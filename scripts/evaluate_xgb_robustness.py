                      

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from xgb_protocol_ids_utils import (
    choose_default_protocol,
    discover_latest_hpo_run,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    prepare_feature_matrix,
    protocol_slug,
    routed_predict,
)


VALID_ATTACK_METHODS = ("surrogate_fgsm", "surrogate_pgd", "heuristic_shap")
VALID_REALISTIC_METHODS = ("query_sparse_hillclimb", "query_sparse_hillclimb_benign")

SEMANTIC_LOCK_FEATURES = {
    "Protocol Type",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
}


def log_progress(message: str, *, start_ts: float | None = None) -> None:
    if start_ts is None:
        print(f"[PROGRESS] {message}", flush=True)
        return
    elapsed = time.time() - start_ts
    print(f"[PROGRESS +{elapsed:8.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to full_gpu_hpo_models_* directory (defaults to latest under reports).",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/merged/metadata_train.csv",
        help="Path to train CSV used for constraint estimation and surrogate training.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/merged/metadata_test.csv",
        help="Path to test CSV used for robustness evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to <run-dir>/xgb_robustness).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0,0.01,0.02,0.05,0.10",
        help="Comma-separated L_inf radii in surrogate-normalized space.",
    )
    parser.add_argument(
        "--attack-methods",
        type=str,
        default="surrogate_fgsm,surrogate_pgd,heuristic_shap",
        help="Comma-separated subset of: surrogate_fgsm,surrogate_pgd,heuristic_shap",
    )
    parser.add_argument(
        "--sample-attack-per-protocol",
        type=int,
        default=25000,
        help="Malicious evaluation rows sampled per protocol from test CSV.",
    )
    parser.add_argument(
        "--sample-benign-per-protocol",
        type=int,
        default=25000,
        help="Benign evaluation rows sampled per protocol from test CSV.",
    )
    parser.add_argument(
        "--surrogate-train-per-protocol",
        type=int,
        default=200000,
        help="Train rows sampled per protocol for surrogate training and constraints.",
    )
    parser.add_argument("--surrogate-epochs", type=int, default=12)
    parser.add_argument("--surrogate-lr", type=float, default=0.08)
    parser.add_argument("--surrogate-batch-size", type=int, default=4096)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha-ratio", type=float, default=0.25)
    parser.add_argument("--top-shap-features", type=int, default=5)
    parser.add_argument("--percentile-lower", type=float, default=1.0)
    parser.add_argument("--percentile-upper", type=float, default=99.0)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument(
        "--xgb-device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for XGBoost inference/scoring in this script.",
    )
    parser.add_argument(
        "--realistic-mode",
        type=str,
        choices=("off", "on"),
        default="off",
        help="Enable query-limited realistic black-box robustness evaluation.",
    )
    parser.add_argument(
        "--query-budget-malicious",
        type=int,
        default=300,
        help="Per-sample query budget for malicious evasion attack.",
    )
    parser.add_argument(
        "--query-budget-benign",
        type=int,
        default=150,
        help="Per-sample query budget for benign-side-effect attack.",
    )
    parser.add_argument(
        "--query-max-steps",
        type=int,
        default=80,
        help="Max hillclimb iterations per sample for query attack.",
    )
    parser.add_argument(
        "--query-candidates-per-step",
        type=int,
        default=4,
        help="Candidate proposals evaluated per hillclimb step.",
    )
    parser.add_argument(
        "--query-feature-subset-size",
        type=int,
        default=3,
        help="Max mutable feature count perturbed per hillclimb step (sparse updates).",
    )
    parser.add_argument(
        "--query-score-batch-rows",
        type=int,
        default=64,
        help="Rows processed together in each hillclimb scoring batch (larger improves GPU utilization).",
    )
    parser.add_argument(
        "--query-max-active-rows-per-step",
        type=int,
        default=0,
        help="Optional cap on active rows processed per hillclimb step (<=0 disables cap).",
    )
    parser.add_argument(
        "--query-stagnation-patience",
        type=int,
        default=0,
        help="Optional early-stop patience in hillclimb steps when margin improvement is below threshold (<=0 disables).",
    )
    parser.add_argument(
        "--query-stagnation-min-delta",
        type=float,
        default=1e-6,
        help="Minimum per-step best-margin improvement to reset hillclimb stagnation counter.",
    )
    parser.add_argument(
        "--query-fast-projection",
        dest="query_fast_projection",
        action="store_true",
        default=True,
        help="Enable fast batched proposal projection with top-k full-refine scoring (reduces CPU bottleneck).",
    )
    parser.add_argument(
        "--no-query-fast-projection",
        dest="query_fast_projection",
        action="store_false",
        help="Disable fast batched projection path and use full projection on all candidates.",
    )
    parser.add_argument(
        "--query-refine-topk",
        type=int,
        default=2,
        help="Number of top fast-projected candidates per row to fully project+rescore.",
    )
    parser.add_argument(
        "--relative-cap-default",
        type=float,
        default=0.20,
        help="Default per-feature relative perturbation cap multiplier.",
    )
    parser.add_argument(
        "--relative-cap-rate-group",
        type=float,
        default=None,
        help="Optional relative cap override for features containing 'rate'.",
    )
    parser.add_argument(
        "--relative-cap-size-group",
        type=float,
        default=None,
        help="Optional relative cap override for size/count-like features.",
    )
    parser.add_argument(
        "--relative-cap-time-group",
        type=float,
        default=None,
        help="Optional relative cap override for time/duration-like features.",
    )
    parser.add_argument(
        "--relation-quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile (0..1) for relation-band constraints.",
    )
    parser.add_argument(
        "--relation-quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile (0..1) for relation-band constraints.",
    )
    parser.add_argument(
        "--realistic-sample-attack-per-protocol",
        type=int,
        default=5000,
        help="Malicious sample cap per protocol for realistic query campaign.",
    )
    parser.add_argument(
        "--realistic-sample-benign-per-protocol",
        type=int,
        default=5000,
        help="Benign sample cap per protocol for realistic query campaign.",
    )
    return parser.parse_args()


def parse_epsilons(raw: str) -> List[float]:
    eps: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0:
            raise ValueError(f"epsilon must be >= 0; got {value}")
        eps.append(float(value))
    if not eps:
        raise ValueError("No epsilon values provided.")
    out: List[float] = []
    seen = set()
    for value in eps:
        key = round(value, 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def parse_attack_methods(raw: str) -> List[str]:
    methods: List[str] = []
    for token in str(raw).split(","):
        name = token.strip()
        if not name:
            continue
        if name not in VALID_ATTACK_METHODS:
            raise ValueError(f"Unknown attack method '{name}'. Valid: {','.join(VALID_ATTACK_METHODS)}")
        methods.append(name)
    if not methods:
        raise ValueError("No attack methods provided.")
    return methods


def parse_realistic_mode(raw: str) -> bool:
    token = str(raw).strip().lower()
    if token not in {"off", "on"}:
        raise ValueError(f"Invalid --realistic-mode value '{raw}'. Use 'off' or 'on'.")
    return token == "on"


def validate_unit_quantile(value: float, name: str) -> float:
    q = float(value)
    if not np.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError(f"{name} must be within [0,1]; got {value}")
    return q


def validate_nonnegative(value: float, name: str) -> float:
    x = float(value)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError(f"{name} must be >= 0; got {value}")
    return x


def feature_name_key(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _set_boosters_device(models_by_protocol: Dict[str, xgb.Booster], device: str) -> None:
    target = str(device).strip().lower()
    if target == "cuda":
        params = {"device": "cuda", "predictor": "gpu_predictor"}
    else:
        params = {"device": "cpu", "predictor": "cpu_predictor"}
    for booster in models_by_protocol.values():
        booster.set_param(params)


def configure_boosters_device(
    models_by_protocol: Dict[str, xgb.Booster],
    feature_count: int,
    requested_device: str,
    start_ts: float,
) -> str:
    target = str(requested_device).strip().lower()
    if target not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported --xgb-device '{requested_device}'.")

    def _can_predict_now() -> bool:
        first = next(iter(models_by_protocol.values()))
        dummy = np.zeros((1, max(1, int(feature_count))), dtype=np.float32)
        try:
            _ = first.predict(xgb.DMatrix(dummy))
            return True
        except Exception:
            return False

    if target == "cpu":
        _set_boosters_device(models_by_protocol, "cpu")
        return "cpu"
    if target == "cuda":
        _set_boosters_device(models_by_protocol, "cuda")
        if not _can_predict_now():
            raise RuntimeError("Requested --xgb-device=cuda, but CUDA inference test failed.")
        return "cuda"

    _set_boosters_device(models_by_protocol, "cuda")
    if _can_predict_now():
        return "cuda"
    log_progress("CUDA inference unavailable for this run; falling back to CPU predictor.", start_ts=start_ts)
    _set_boosters_device(models_by_protocol, "cpu")
    return "cpu"


def normalize_protocol_series(series: pd.Series) -> np.ndarray:
    return series.fillna("unknown").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)


def normalize_label_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_i = y_true.astype(np.int8, copy=False)
    y_pred_i = y_pred.astype(np.int8, copy=False)

    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "n": int(len(y_true_i)),
        "n_benign": int(np.sum(y_true_i == 0)),
        "n_attack": int(np.sum(y_true_i == 1)),
    }


def compute_asr(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    adv_pred: np.ndarray,
    scope_mask: np.ndarray,
) -> Dict[str, float]:
    attack_mask = (y_true == 1) & scope_mask
    cond_mask = attack_mask & (baseline_pred == 1)
    flip_cond = int(np.sum(cond_mask & (adv_pred == 0)))
    cond_total = int(np.sum(cond_mask))
    all_total = int(np.sum(attack_mask))
    benign_pred_total = int(np.sum(attack_mask & (adv_pred == 0)))

    asr_cond = float(flip_cond / cond_total) if cond_total > 0 else float("nan")
    asr_all = float(benign_pred_total / all_total) if all_total > 0 else float("nan")
    return {
        "asr_conditional": asr_cond,
        "asr_conditional_num": flip_cond,
        "asr_conditional_den": cond_total,
        "asr_all": asr_all,
        "asr_all_num": benign_pred_total,
        "asr_all_den": all_total,
    }


def summary_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
    }


def ensure_protocols_present(counts: Dict[str, int], protocols: Sequence[str], what: str) -> None:
    missing = [p for p in protocols if counts.get(p, 0) <= 0]
    if missing:
        raise RuntimeError(f"Missing {what} rows for protocols: {missing}")


def count_train_rows_by_protocol(
    train_csv: Path,
    chunk_size: int,
    start_ts: float,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total = 0
    reader = pd.read_csv(
        train_csv,
        usecols=["protocol_hint"],
        chunksize=max(1, int(chunk_size)),
    )
    for i, chunk in enumerate(reader, start=1):
        proto = normalize_protocol_series(chunk["protocol_hint"])
        vc = pd.Series(proto).value_counts()
        for key, value in vc.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
        total += len(chunk)
        if i % 5 == 0:
            log_progress(f"train count pass: chunks={i}, rows={total}", start_ts=start_ts)
    log_progress(f"train count pass complete: rows={total}", start_ts=start_ts)
    return counts


def count_test_rows_by_protocol_label(
    test_csv: Path,
    chunk_size: int,
    start_ts: float,
) -> Dict[Tuple[str, int], int]:
    counts: Dict[Tuple[str, int], int] = {}
    total = 0
    reader = pd.read_csv(
        test_csv,
        usecols=["protocol_hint", "label"],
        chunksize=max(1, int(chunk_size)),
    )
    for i, chunk in enumerate(reader, start=1):
        proto = normalize_protocol_series(chunk["protocol_hint"])
        label = normalize_label_series(chunk["label"])
        tmp = pd.DataFrame({"protocol_hint": proto, "label": label})
        grouped = tmp.groupby(["protocol_hint", "label"]).size()
        for (p, y), value in grouped.items():
            key = (str(p), int(y))
            counts[key] = counts.get(key, 0) + int(value)
        total += len(chunk)
        if i % 5 == 0:
            log_progress(f"test count pass: chunks={i}, rows={total}", start_ts=start_ts)
    log_progress(f"test count pass complete: rows={total}", start_ts=start_ts)
    return counts


def choose_positions(
    counts: Dict[Any, int],
    targets: Dict[Any, int],
    rng: np.random.Generator,
) -> Dict[Any, np.ndarray]:
    out: Dict[Any, np.ndarray] = {}
    for key, total in counts.items():
        want = int(max(0, min(int(targets.get(key, 0)), int(total))))
        if want <= 0:
            out[key] = np.empty(0, dtype=np.int64)
        elif want >= int(total):
            out[key] = np.arange(int(total), dtype=np.int64)
        else:
            picked = rng.choice(int(total), size=want, replace=False).astype(np.int64, copy=False)
            picked.sort()
            out[key] = picked
    return out


def sample_train_rows_by_protocol(
    train_csv: Path,
    feature_columns: List[str],
    target_per_protocol: int,
    protocols: Sequence[str],
    seed: int,
    chunk_size: int,
    start_ts: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    counts = count_train_rows_by_protocol(train_csv, chunk_size=chunk_size, start_ts=start_ts)
    ensure_protocols_present(counts, protocols, "train")

    targets = {proto: int(target_per_protocol) for proto in counts.keys()}
    rng = np.random.default_rng(seed)
    selected_positions = choose_positions(counts=counts, targets=targets, rng=rng)

    offsets = {proto: 0 for proto in counts.keys()}
    pieces: List[pd.DataFrame] = []
    usecols = ["protocol_hint", "label"] + feature_columns

    total = 0
    reader = pd.read_csv(
        train_csv,
        usecols=usecols,
        chunksize=max(1, int(chunk_size)),
    )
    for i, chunk in enumerate(reader, start=1):
        proto_arr = normalize_protocol_series(chunk["protocol_hint"])
        chunk["protocol_hint"] = proto_arr
        chunk["label"] = normalize_label_series(chunk["label"])
        total += len(chunk)

        for proto in counts.keys():
            idx_proto = np.flatnonzero(proto_arr == proto)
            n = int(idx_proto.size)
            if n <= 0:
                continue

            start = int(offsets[proto])
            end = start + n
            sel = selected_positions.get(proto, np.empty(0, dtype=np.int64))
            if sel.size > 0:
                left = int(np.searchsorted(sel, start, side="left"))
                right = int(np.searchsorted(sel, end, side="left"))
                if right > left:
                    rel = sel[left:right] - start
                    picked_idx = idx_proto[rel]
                    pieces.append(chunk.iloc[picked_idx].copy())
            offsets[proto] = end
        if i % 5 == 0:
            log_progress(f"train sample pass: chunks={i}, rows={total}", start_ts=start_ts)

    if not pieces:
        raise RuntimeError("Train sampling produced no rows.")
    sampled = pd.concat(pieces, ignore_index=True)
    sampled["protocol_hint"] = sampled["protocol_hint"].astype(str).map(protocol_slug)
    sampled["label"] = pd.to_numeric(sampled["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1)

    by_protocol = sampled["protocol_hint"].value_counts().to_dict()
    meta = {
        "counts_available": {k: int(v) for k, v in counts.items()},
        "counts_sampled": {str(k): int(v) for k, v in by_protocol.items()},
        "target_per_protocol": int(target_per_protocol),
        "rows_sampled_total": int(len(sampled)),
    }
    log_progress(f"train sampling complete: sampled_rows={len(sampled)}", start_ts=start_ts)
    return sampled, meta


def sample_test_rows_balanced(
    test_csv: Path,
    feature_columns: List[str],
    sample_attack_per_protocol: int,
    sample_benign_per_protocol: int,
    protocols: Sequence[str],
    seed: int,
    chunk_size: int,
    start_ts: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    counts = count_test_rows_by_protocol_label(test_csv, chunk_size=chunk_size, start_ts=start_ts)
    targets: Dict[Tuple[str, int], int] = {}
    for key in counts.keys():
        _, label = key
        targets[key] = int(sample_attack_per_protocol if label == 1 else sample_benign_per_protocol)

    rng = np.random.default_rng(seed)
    selected_positions = choose_positions(counts=counts, targets=targets, rng=rng)
    offsets = {key: 0 for key in counts.keys()}

    pieces: List[pd.DataFrame] = []
    usecols = ["protocol_hint", "label"] + feature_columns
    total = 0
    reader = pd.read_csv(
        test_csv,
        usecols=usecols,
        chunksize=max(1, int(chunk_size)),
    )
    for i, chunk in enumerate(reader, start=1):
        proto_arr = normalize_protocol_series(chunk["protocol_hint"])
        label_arr = normalize_label_series(chunk["label"])
        chunk["protocol_hint"] = proto_arr
        chunk["label"] = label_arr
        total += len(chunk)

        for key in counts.keys():
            proto, label = key
            idx = np.flatnonzero((proto_arr == proto) & (label_arr == label))
            n = int(idx.size)
            if n <= 0:
                continue

            start = int(offsets[key])
            end = start + n
            sel = selected_positions.get(key, np.empty(0, dtype=np.int64))
            if sel.size > 0:
                left = int(np.searchsorted(sel, start, side="left"))
                right = int(np.searchsorted(sel, end, side="left"))
                if right > left:
                    rel = sel[left:right] - start
                    picked_idx = idx[rel]
                    pieces.append(chunk.iloc[picked_idx].copy())
            offsets[key] = end
        if i % 5 == 0:
            log_progress(f"test sample pass: chunks={i}, rows={total}", start_ts=start_ts)

    if not pieces:
        raise RuntimeError("Balanced test sampling produced no rows.")
    sampled = pd.concat(pieces, ignore_index=True)
    sampled["protocol_hint"] = sampled["protocol_hint"].astype(str).map(protocol_slug)
    sampled["label"] = pd.to_numeric(sampled["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1)

    sampled_counts = sampled.groupby(["protocol_hint", "label"]).size().to_dict() if not sampled.empty else {}
    by_proto_total = sampled["protocol_hint"].value_counts().to_dict()
    missing_protocols = [p for p in protocols if int(by_proto_total.get(p, 0)) <= 0]
    if missing_protocols:
        raise RuntimeError(f"No sampled evaluation rows for protocols: {missing_protocols}")

    meta = {
        "counts_available": {f"{k[0]}::{k[1]}": int(v) for k, v in counts.items()},
        "counts_sampled": {f"{k[0]}::{k[1]}": int(v) for k, v in sampled_counts.items()},
        "sample_attack_per_protocol": int(sample_attack_per_protocol),
        "sample_benign_per_protocol": int(sample_benign_per_protocol),
        "rows_sampled_total": int(len(sampled)),
    }
    log_progress(f"test sampling complete: sampled_rows={len(sampled)}", start_ts=start_ts)
    return sampled, meta


def build_constraints(
    train_sample_df: pd.DataFrame,
    feature_columns: List[str],
    protocols: Sequence[str],
    percentile_lower: float,
    percentile_upper: float,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    constraints: Dict[str, Dict[str, np.ndarray]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for proto in protocols:
        proto_df = train_sample_df.loc[train_sample_df["protocol_hint"] == proto].copy()
        if proto_df.empty:
            raise RuntimeError(f"Cannot build constraints: no train samples for protocol '{proto}'")
        x = prepare_feature_matrix(proto_df.copy(), feature_columns).astype(np.float64, copy=False)
        n_rows, n_features = x.shape
        lower = np.zeros(n_features, dtype=np.float32)
        upper = np.zeros(n_features, dtype=np.float32)
        locked_mask = np.zeros(n_features, dtype=bool)

        for j, feature in enumerate(feature_columns):
            col = x[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                p_lo = 0.0
                p_hi = 0.0
                raw_min = 0.0
                raw_max = 0.0
                uniq = np.array([0.0], dtype=np.float64)
            else:
                p_lo = float(np.percentile(finite, percentile_lower))
                p_hi = float(np.percentile(finite, percentile_upper))
                raw_min = float(np.min(finite))
                raw_max = float(np.max(finite))
                uniq = np.unique(np.round(finite, 6))

            lo = max(0.0, p_lo)
            hi = float(p_hi)
            if not np.isfinite(hi):
                hi = lo
            if hi < lo:
                hi = lo

            is_semantic_lock = feature in SEMANTIC_LOCK_FEATURES
            low_cardinality_lock = bool(
                uniq.size <= 20
                and float(np.min(uniq)) >= -1e-8
                and float(np.max(uniq)) <= 1.0 + 1e-8
            )
            reasons: List[str] = []
            if is_semantic_lock:
                reasons.append("semantic")
            if low_cardinality_lock:
                reasons.append("auto_low_cardinality")
            is_locked = len(reasons) > 0

            lower[j] = float(lo)
            upper[j] = float(hi)
            locked_mask[j] = bool(is_locked)

            summary_rows.append(
                {
                    "protocol_hint": proto,
                    "feature": feature,
                    "lower_bound": float(lo),
                    "upper_bound": float(hi),
                    "is_locked": int(is_locked),
                    "lock_reason": "|".join(reasons),
                    "sample_rows": int(n_rows),
                    "sample_min": raw_min,
                    "sample_max": raw_max,
                    "sample_p_low": float(p_lo),
                    "sample_p_high": float(p_hi),
                    "sample_unique_count": int(uniq.size),
                }
            )

        span = np.maximum(upper.astype(np.float64) - lower.astype(np.float64), 1e-12).astype(np.float32)
        constraints[proto] = {
            "lower": lower.astype(np.float32, copy=False),
            "upper": upper.astype(np.float32, copy=False),
            "locked_mask": locked_mask,
            "locked_idx": np.flatnonzero(locked_mask).astype(np.int64),
            "span": span,
        }

    summary_df = pd.DataFrame(summary_rows)
    return constraints, summary_df


def train_numpy_logistic_surrogate(
    x_raw: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    progress_prefix: str,
    start_ts: float,
) -> Dict[str, Any]:
    x = np.asarray(x_raw, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    n_rows, n_features = x.shape
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    x_norm = (x - mean) / std

    n_pos = int(np.sum(y_vec == 1.0))
    n_neg = int(np.sum(y_vec == 0.0))
    if n_pos == 0 or n_neg == 0:
        log_progress(f"{progress_prefix}: single-class surrogate (n_pos={n_pos}, n_neg={n_neg})", start_ts=start_ts)
        return {
            "w": np.zeros(n_features, dtype=np.float64),
            "b": 0.0,
            "mean": mean.astype(np.float64),
            "std": std.astype(np.float64),
            "n_rows": int(n_rows),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "single_class": 1,
        }

    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    weight_pos = float(n_rows / (2.0 * n_pos))
    weight_neg = float(n_rows / (2.0 * n_neg))
    rng = np.random.default_rng(seed)
    bs = max(1, int(batch_size))
    n_epochs = max(1, int(epochs))

    for epoch in range(1, n_epochs + 1):
        order = rng.permutation(n_rows)
        running_loss = 0.0
        seen = 0
        for start in range(0, n_rows, bs):
            idx = order[start : start + bs]
            xb = x_norm[idx]
            yb = y_vec[idx]
            z = xb @ w + b
            p = stable_sigmoid(z)
            sample_weight = np.where(yb == 1.0, weight_pos, weight_neg)
            error = (p - yb) * sample_weight
            grad_w = (xb.T @ error) / float(len(idx))
            grad_b = float(np.mean(error))
            w -= float(lr) * grad_w
            b -= float(lr) * grad_b

            p_clip = np.clip(p, 1e-8, 1.0 - 1e-8)
            bce = -(yb * np.log(p_clip) + (1.0 - yb) * np.log(1.0 - p_clip))
            running_loss += float(np.sum(bce))
            seen += len(idx)
        log_progress(
            f"{progress_prefix}: epoch {epoch}/{n_epochs}, mean_bce={running_loss / max(1, seen):.6f}",
            start_ts=start_ts,
        )

    return {
        "w": w,
        "b": float(b),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "n_rows": int(n_rows),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "single_class": 0,
    }


def train_surrogates_by_protocol(
    train_sample_df: pd.DataFrame,
    feature_columns: List[str],
    protocols: Sequence[str],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    start_ts: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    surrogates: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    for i, proto in enumerate(protocols):
        proto_df = train_sample_df.loc[train_sample_df["protocol_hint"] == proto].copy()
        if proto_df.empty:
            raise RuntimeError(f"Cannot train surrogate: no train sample rows for protocol '{proto}'")
        x = prepare_feature_matrix(proto_df.copy(), feature_columns).astype(np.float64, copy=False)
        y = pd.to_numeric(proto_df["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
        surrogate = train_numpy_logistic_surrogate(
            x_raw=x,
            y=y,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=int(seed + 1009 + i * 17),
            progress_prefix=f"surrogate[{proto}]",
            start_ts=start_ts,
        )
        surrogates[proto] = surrogate
        meta[proto] = {
            "n_rows": int(surrogate["n_rows"]),
            "n_pos": int(surrogate["n_pos"]),
            "n_neg": int(surrogate["n_neg"]),
            "single_class": int(surrogate["single_class"]),
        }
    return surrogates, meta


def project_constrained(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    locked_idx: np.ndarray,
) -> np.ndarray:
    out = np.clip(x_adv, lower, upper)
    if locked_idx.size > 0:
        out[:, locked_idx] = x_orig[:, locked_idx]
    return out


def attack_surrogate_fgsm(
    x_orig: np.ndarray,
    surrogate: Dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    locked_idx: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0.0:
        return x_orig.copy()
    w = np.asarray(surrogate["w"], dtype=np.float64)
    b = float(surrogate["b"])
    mean = np.asarray(surrogate["mean"], dtype=np.float64)
    std = np.asarray(surrogate["std"], dtype=np.float64)

    x_norm = (x_orig.astype(np.float64) - mean) / std
    p = stable_sigmoid(x_norm @ w + b)
    grad = (p - 1.0)[:, None] * w[None, :]
    if locked_idx.size > 0:
        grad[:, locked_idx] = 0.0

    x_adv_norm = x_norm + float(epsilon) * np.sign(grad)
    x_adv_raw = x_adv_norm * std + mean
    x_adv_raw = project_constrained(x_adv_raw, x_orig.astype(np.float64), lower, upper, locked_idx)
    return x_adv_raw.astype(np.float32, copy=False)


def attack_surrogate_pgd(
    x_orig: np.ndarray,
    surrogate: Dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    locked_idx: np.ndarray,
    epsilon: float,
    steps: int,
    alpha_ratio: float,
) -> np.ndarray:
    if epsilon <= 0.0:
        return x_orig.copy()
    n_steps = max(1, int(steps))
    alpha = float(max(0.0, alpha_ratio) * epsilon)
    if alpha <= 0.0:
        alpha = float(epsilon / max(1, n_steps))

    w = np.asarray(surrogate["w"], dtype=np.float64)
    b = float(surrogate["b"])
    mean = np.asarray(surrogate["mean"], dtype=np.float64)
    std = np.asarray(surrogate["std"], dtype=np.float64)

    x0 = x_orig.astype(np.float64)
    x0_norm = (x0 - mean) / std
    x_adv_norm = x0_norm.copy()

    for _ in range(n_steps):
        p = stable_sigmoid(x_adv_norm @ w + b)
        grad = (p - 1.0)[:, None] * w[None, :]
        if locked_idx.size > 0:
            grad[:, locked_idx] = 0.0
        x_adv_norm = x_adv_norm + alpha * np.sign(grad)

        delta = np.clip(x_adv_norm - x0_norm, -epsilon, epsilon)
        if locked_idx.size > 0:
            delta[:, locked_idx] = 0.0
        x_adv_norm = x0_norm + delta

        x_adv_raw = x_adv_norm * std + mean
        x_adv_raw = project_constrained(x_adv_raw, x0, lower, upper, locked_idx)
        x_adv_norm = (x_adv_raw - mean) / std

    x_adv = x_adv_norm * std + mean
    x_adv = project_constrained(x_adv, x0, lower, upper, locked_idx)
    return x_adv.astype(np.float32, copy=False)


def choose_top_shap_features(
    booster: xgb.Booster,
    x_malicious: np.ndarray,
    feature_columns: List[str],
    locked_mask: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    if x_malicious.size == 0:
        return {"indices": [], "features": [], "mean_contrib": {}}
    contrib = booster.predict(xgb.DMatrix(x_malicious), pred_contribs=True)
    contrib_no_bias = contrib[:, :-1]
    mean_contrib = contrib_no_bias.mean(axis=0)

    candidate_idx = np.argsort(mean_contrib)[::-1]
    picked: List[int] = []
    for idx in candidate_idx.tolist():
        if int(top_k) > 0 and len(picked) >= int(top_k):
            break
        if bool(locked_mask[idx]):
            continue
        if float(mean_contrib[idx]) <= 0.0:
            continue
        picked.append(int(idx))

    return {
        "indices": picked,
        "features": [feature_columns[i] for i in picked],
        "mean_contrib": {feature_columns[i]: float(mean_contrib[i]) for i in picked},
    }


def attack_shap_heuristic(
    x_orig: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    locked_idx: np.ndarray,
    top_indices: Sequence[int],
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0.0 or len(top_indices) == 0:
        return x_orig.copy()
    x_adv = x_orig.astype(np.float64).copy()
    span = np.maximum(upper - lower, 0.0)
    for idx in top_indices:
        j = int(idx)
        x_adv[:, j] = x_adv[:, j] - float(epsilon) * float(span[j])
    x_adv = project_constrained(x_adv, x_orig.astype(np.float64), lower, upper, locked_idx)
    return x_adv.astype(np.float32, copy=False)


def score_routed_predictions(
    x_matrix: np.ndarray,
    protocol_hint: np.ndarray,
    feature_columns: List[str],
    models_by_protocol: Dict[str, xgb.Booster],
    thresholds_by_protocol: Dict[str, float],
    default_protocol: str,
) -> pd.DataFrame:
    df = pd.DataFrame(x_matrix, columns=feature_columns)
    df["protocol_hint"] = protocol_hint
    return routed_predict(
        df=df,
        feature_columns=feature_columns,
        models_by_protocol=models_by_protocol,
        thresholds_by_protocol=thresholds_by_protocol,
        default_protocol=default_protocol,
    )


def build_rowwise_tensor(
    protocol_used: np.ndarray,
    protocol_to_vector: Dict[str, np.ndarray],
) -> np.ndarray:
    n = len(protocol_used)
    if n == 0:
        first = next(iter(protocol_to_vector.values()))
        return np.empty((0, len(first)), dtype=np.float32)
    d = len(next(iter(protocol_to_vector.values())))
    out = np.zeros((n, d), dtype=np.float32)
    for proto, vec in protocol_to_vector.items():
        mask = protocol_used == proto
        if np.any(mask):
            out[mask] = vec
    return out


def summarize_perturbations(
    x_base_mal: np.ndarray,
    x_adv_mal: np.ndarray,
    span_mal: np.ndarray,
) -> Dict[str, float]:
    if x_base_mal.size == 0:
        return {
            "rows": 0,
            "l0_mean": float("nan"),
            "l0_median": float("nan"),
            "l0_p95": float("nan"),
            "l2_mean": float("nan"),
            "l2_median": float("nan"),
            "l2_p95": float("nan"),
            "linf_mean": float("nan"),
            "linf_median": float("nan"),
            "linf_p95": float("nan"),
            "l2_norm_mean": float("nan"),
            "l2_norm_median": float("nan"),
            "l2_norm_p95": float("nan"),
            "linf_norm_mean": float("nan"),
            "linf_norm_median": float("nan"),
            "linf_norm_p95": float("nan"),
        }

    delta = x_adv_mal.astype(np.float64) - x_base_mal.astype(np.float64)
    abs_delta = np.abs(delta)
    l0 = np.count_nonzero(abs_delta > 1e-12, axis=1).astype(np.float64)
    l2 = np.linalg.norm(delta, ord=2, axis=1)
    linf = np.max(abs_delta, axis=1)

    span_safe = np.where(span_mal.astype(np.float64) > 1e-12, span_mal.astype(np.float64), 1.0)
    delta_norm = delta / span_safe
    abs_delta_norm = np.abs(delta_norm)
    l2_norm = np.linalg.norm(delta_norm, ord=2, axis=1)
    linf_norm = np.max(abs_delta_norm, axis=1)

    out = {"rows": int(delta.shape[0])}
    out.update({f"l0_{k}": v for k, v in summary_stats(l0).items()})
    out.update({f"l2_{k}": v for k, v in summary_stats(l2).items()})
    out.update({f"linf_{k}": v for k, v in summary_stats(linf).items()})
    out.update({f"l2_norm_{k}": v for k, v in summary_stats(l2_norm).items()})
    out.update({f"linf_norm_{k}": v for k, v in summary_stats(linf_norm).items()})
    return out


def find_feature_index(
    feature_columns: Sequence[str],
    exact_aliases: Sequence[str],
    key_aliases: Sequence[str],
) -> Optional[int]:
    direct: Dict[str, int] = {}
    keyed: Dict[str, int] = {}
    for idx, name in enumerate(feature_columns):
        direct.setdefault(str(name).strip().lower(), int(idx))
        keyed.setdefault(feature_name_key(str(name)), int(idx))

    for alias in exact_aliases:
        idx = direct.get(str(alias).strip().lower())
        if idx is not None:
            return int(idx)
    for alias in key_aliases:
        idx = keyed.get(feature_name_key(alias))
        if idx is not None:
            return int(idx)
    return None


def resolve_relation_feature_indices(feature_columns: Sequence[str]) -> Dict[str, Optional[int]]:
    idx_min = find_feature_index(feature_columns, exact_aliases=("Min",), key_aliases=("min", "minimum"))
    idx_avg = find_feature_index(feature_columns, exact_aliases=("AVG", "Avg", "Average"), key_aliases=("avg", "average", "mean"))
    idx_max = find_feature_index(feature_columns, exact_aliases=("Max",), key_aliases=("max", "maximum"))
    idx_tot = find_feature_index(
        feature_columns,
        exact_aliases=("Tot sum", "Total sum"),
        key_aliases=("totsum", "totalsum", "sum"),
    )
    idx_num = find_feature_index(feature_columns, exact_aliases=("Number",), key_aliases=("number", "count", "num"))
    idx_srate = find_feature_index(feature_columns, exact_aliases=("Srate",), key_aliases=("srate", "srcrate"))
    idx_drate = find_feature_index(feature_columns, exact_aliases=("Drate",), key_aliases=("drate", "dstrate"))
    idx_rate = find_feature_index(feature_columns, exact_aliases=("Rate",), key_aliases=("rate",))
    return {
        "min": idx_min,
        "avg": idx_avg,
        "max": idx_max,
        "tot_sum": idx_tot,
        "number": idx_num,
        "srate": idx_srate,
        "drate": idx_drate,
        "rate": idx_rate,
    }


def compute_quantile_band(
    values: np.ndarray,
    q_low: float,
    q_high: float,
    *,
    nonnegative: bool = False,
) -> Tuple[Optional[float], Optional[float], int]:
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n <= 0:
        return None, None, 0
    low = float(np.quantile(finite, q_low))
    high = float(np.quantile(finite, q_high))
    if high < low:
        low, high = high, low
    width = max(high - low, 1e-9)
    pad = 0.05 * width + 1e-9
    low_adj = low - pad
    high_adj = high + pad
    if nonnegative:
        low_adj = max(0.0, low_adj)
    return float(low_adj), float(high_adj), n


def build_relative_caps_vector(
    feature_columns: Sequence[str],
    *,
    default_cap: float,
    rate_group_cap: Optional[float],
    size_group_cap: Optional[float],
    time_group_cap: Optional[float],
) -> np.ndarray:
    caps = np.full(len(feature_columns), max(0.0, float(default_cap)), dtype=np.float64)
    for j, feature in enumerate(feature_columns):
        name = str(feature).lower()
        key = feature_name_key(feature)
        candidates: List[float] = [caps[j]]

        if rate_group_cap is not None and ("rate" in name or "rate" in key):
            candidates.append(max(0.0, float(rate_group_cap)))
        if size_group_cap is not None and any(tok in key for tok in ("byte", "packet", "count", "number", "size", "tot", "sum", "len", "volume")):
            candidates.append(max(0.0, float(size_group_cap)))
        if time_group_cap is not None and any(tok in key for tok in ("time", "dur", "iat", "interval", "latency")):
            candidates.append(max(0.0, float(time_group_cap)))
        caps[j] = float(min(candidates))
    return caps


def build_realism_profiles(
    train_sample_df: pd.DataFrame,
    feature_columns: List[str],
    protocols: Sequence[str],
    constraints: Dict[str, Dict[str, np.ndarray]],
    relation_q_low: float,
    relation_q_high: float,
    relative_cap_default: float,
    relative_cap_rate_group: Optional[float],
    relative_cap_size_group: Optional[float],
    relative_cap_time_group: Optional[float],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    relation_idx = resolve_relation_feature_indices(feature_columns)
    relative_caps_global = build_relative_caps_vector(
        feature_columns,
        default_cap=float(relative_cap_default),
        rate_group_cap=relative_cap_rate_group,
        size_group_cap=relative_cap_size_group,
        time_group_cap=relative_cap_time_group,
    )
    profiles: Dict[str, Dict[str, Any]] = {}
    profile_json: Dict[str, Any] = {
        "relation_quantile_low": float(relation_q_low),
        "relation_quantile_high": float(relation_q_high),
        "relation_feature_indices": {k: (None if v is None else int(v)) for k, v in relation_idx.items()},
        "relative_cap": {
            "default": float(relative_cap_default),
            "rate_group": None if relative_cap_rate_group is None else float(relative_cap_rate_group),
            "size_group": None if relative_cap_size_group is None else float(relative_cap_size_group),
            "time_group": None if relative_cap_time_group is None else float(relative_cap_time_group),
        },
        "protocols": {},
    }

    for proto in protocols:
        proto_df = train_sample_df.loc[train_sample_df["protocol_hint"] == proto].copy()
        if proto_df.empty:
            raise RuntimeError(f"Cannot build realism profile: no train rows for protocol '{proto}'")
        x = prepare_feature_matrix(proto_df.copy(), feature_columns).astype(np.float64, copy=False)
        lower = constraints[proto]["lower"].astype(np.float64)
        upper = constraints[proto]["upper"].astype(np.float64)
        locked_mask = constraints[proto]["locked_mask"].astype(bool, copy=False)
        locked_idx = constraints[proto]["locked_idx"].astype(np.int64, copy=False)
        span = constraints[proto]["span"].astype(np.float64, copy=False)

        proto_relative_caps = relative_caps_global.copy()
        mutable_mask = (~locked_mask) & (span > 1e-12) & (proto_relative_caps > 0.0)
        mutable_idx = np.flatnonzero(mutable_mask).astype(np.int64)

        idx_tot = relation_idx["tot_sum"]
        idx_num = relation_idx["number"]
        idx_avg = relation_idx["avg"]
        if idx_tot is not None and idx_num is not None and idx_avg is not None:
            denom = x[:, idx_num] * x[:, idx_avg]
            valid = np.isfinite(denom) & (denom > 1e-9) & np.isfinite(x[:, idx_tot])
            ratio_vals = np.divide(
                x[valid, idx_tot],
                denom[valid],
                out=np.zeros(np.sum(valid), dtype=np.float64),
                where=denom[valid] > 0,
            )
            r1_low, r1_high, r1_n = compute_quantile_band(
                ratio_vals,
                q_low=float(relation_q_low),
                q_high=float(relation_q_high),
                nonnegative=True,
            )
        else:
            r1_low, r1_high, r1_n = None, None, 0

        idx_srate = relation_idx["srate"]
        idx_drate = relation_idx["drate"]
        idx_rate = relation_idx["rate"]
        if idx_srate is not None and idx_drate is not None and idx_rate is not None:
            rate = x[:, idx_rate]
            numer = x[:, idx_srate] + x[:, idx_drate]
            valid = np.isfinite(rate) & (rate > 1e-9) & np.isfinite(numer)
            ratio_vals = np.divide(
                numer[valid],
                rate[valid],
                out=np.zeros(np.sum(valid), dtype=np.float64),
                where=rate[valid] > 0,
            )
            r2_low, r2_high, r2_n = compute_quantile_band(
                ratio_vals,
                q_low=float(relation_q_low),
                q_high=float(relation_q_high),
                nonnegative=True,
            )
        else:
            r2_low, r2_high, r2_n = None, None, 0

        min_avg_max_enabled = (
            relation_idx["min"] is not None and relation_idx["avg"] is not None and relation_idx["max"] is not None
        )
        profiles[proto] = {
            "lower": lower,
            "upper": upper,
            "locked_mask": locked_mask,
            "locked_idx": locked_idx,
            "span": span,
            "relative_caps": proto_relative_caps,
            "mutable_idx": mutable_idx,
            "relation_idx": relation_idx.copy(),
            "min_avg_max_enabled": bool(min_avg_max_enabled),
            "ratio_tot_num_avg_enabled": bool(r1_low is not None and r1_high is not None),
            "ratio_tot_num_avg_low": None if r1_low is None else float(r1_low),
            "ratio_tot_num_avg_high": None if r1_high is None else float(r1_high),
            "ratio_tot_num_avg_n": int(r1_n),
            "ratio_rate_enabled": bool(r2_low is not None and r2_high is not None),
            "ratio_rate_low": None if r2_low is None else float(r2_low),
            "ratio_rate_high": None if r2_high is None else float(r2_high),
            "ratio_rate_n": int(r2_n),
        }
        profile_json["protocols"][proto] = {
            "rows_used": int(len(proto_df)),
            "locked_features": int(np.sum(locked_mask)),
            "mutable_features": int(mutable_idx.size),
            "min_avg_max_enabled": bool(min_avg_max_enabled),
            "ratio_tot_num_avg": {
                "enabled": bool(r1_low is not None and r1_high is not None),
                "low": None if r1_low is None else float(r1_low),
                "high": None if r1_high is None else float(r1_high),
                "valid_rows": int(r1_n),
            },
            "ratio_rate": {
                "enabled": bool(r2_low is not None and r2_high is not None),
                "low": None if r2_low is None else float(r2_low),
                "high": None if r2_high is None else float(r2_high),
                "valid_rows": int(r2_n),
            },
        }
    return profiles, profile_json


def compute_row_max_delta(x_orig: np.ndarray, profile: Dict[str, Any], epsilon: float) -> np.ndarray:
    span = profile["span"].astype(np.float64, copy=False)
    rel_caps = profile["relative_caps"].astype(np.float64, copy=False)
    cap_global = max(0.0, float(epsilon)) * span
    cap_relative = rel_caps * np.maximum(np.abs(x_orig), 1.0)
    cap = np.minimum(cap_global, cap_relative)
    locked_idx = profile["locked_idx"]
    if locked_idx.size > 0:
        cap[locked_idx] = 0.0
    return cap


def enforce_relation_constraints_row(x: np.ndarray, profile: Dict[str, Any]) -> None:
    idx = profile["relation_idx"]
    locked_mask = profile["locked_mask"]

    idx_min = idx["min"]
    idx_avg = idx["avg"]
    idx_max = idx["max"]
    if profile["min_avg_max_enabled"] and idx_min is not None and idx_avg is not None and idx_max is not None:
        mn = float(x[idx_min])
        av = float(x[idx_avg])
        mx = float(x[idx_max])

        if mn > mx:
            if not locked_mask[idx_min] and not locked_mask[idx_max]:
                mid = 0.5 * (mn + mx)
                mn = mid
                mx = mid
            elif not locked_mask[idx_min]:
                mn = mx
            elif not locked_mask[idx_max]:
                mx = mn
        if av < mn:
            if not locked_mask[idx_avg]:
                av = mn
            elif not locked_mask[idx_min]:
                mn = av
        elif av > mx:
            if not locked_mask[idx_avg]:
                av = mx
            elif not locked_mask[idx_max]:
                mx = av

        x[idx_min] = mn
        x[idx_avg] = av
        x[idx_max] = mx

    idx_tot = idx["tot_sum"]
    idx_num = idx["number"]
    if (
        profile["ratio_tot_num_avg_enabled"]
        and idx_tot is not None
        and idx_num is not None
        and idx_avg is not None
        and profile["ratio_tot_num_avg_low"] is not None
        and profile["ratio_tot_num_avg_high"] is not None
    ):
        denom = float(x[idx_num]) * float(x[idx_avg])
        if denom > 1e-9:
            lo = float(profile["ratio_tot_num_avg_low"])
            hi = float(profile["ratio_tot_num_avg_high"])
            ratio = float(x[idx_tot]) / denom
            target_ratio = float(np.clip(ratio, lo, hi))
            if not locked_mask[idx_tot]:
                x[idx_tot] = target_ratio * denom
            elif not locked_mask[idx_avg] and float(x[idx_num]) > 1e-9:
                x[idx_avg] = float(x[idx_tot]) / (float(x[idx_num]) * target_ratio)
            elif not locked_mask[idx_num] and float(x[idx_avg]) > 1e-9:
                x[idx_num] = float(x[idx_tot]) / (float(x[idx_avg]) * target_ratio)

    idx_srate = idx["srate"]
    idx_drate = idx["drate"]
    idx_rate = idx["rate"]
    if (
        profile["ratio_rate_enabled"]
        and idx_srate is not None
        and idx_drate is not None
        and idx_rate is not None
        and profile["ratio_rate_low"] is not None
        and profile["ratio_rate_high"] is not None
    ):
        rate = float(x[idx_rate])
        if rate > 1e-9:
            lo = float(profile["ratio_rate_low"])
            hi = float(profile["ratio_rate_high"])
            srate = float(x[idx_srate])
            drate = float(x[idx_drate])
            numer = srate + drate
            ratio = numer / rate
            target_ratio = float(np.clip(ratio, lo, hi))
            target_numer = target_ratio * rate
            if not locked_mask[idx_srate] and not locked_mask[idx_drate]:
                if numer <= 1e-12:
                    x[idx_srate] = 0.5 * target_numer
                    x[idx_drate] = 0.5 * target_numer
                else:
                    scale = target_numer / numer
                    x[idx_srate] = srate * scale
                    x[idx_drate] = drate * scale
            elif not locked_mask[idx_rate] and target_ratio > 1e-12:
                x[idx_rate] = numer / target_ratio


def relation_violation_masks(x_rows: np.ndarray, profile: Dict[str, Any], tol: float = 1e-6) -> Dict[str, np.ndarray]:
    n_rows = x_rows.shape[0]
    none_mask = np.zeros(n_rows, dtype=bool)
    idx = profile["relation_idx"]

    v_min_avg_max = np.zeros(n_rows, dtype=bool)
    if profile["min_avg_max_enabled"] and idx["min"] is not None and idx["avg"] is not None and idx["max"] is not None:
        mn = x_rows[:, idx["min"]]
        av = x_rows[:, idx["avg"]]
        mx = x_rows[:, idx["max"]]
        v_min_avg_max = (av < (mn - tol)) | (av > (mx + tol))

    v_ratio1 = np.zeros(n_rows, dtype=bool)
    if (
        profile["ratio_tot_num_avg_enabled"]
        and idx["tot_sum"] is not None
        and idx["number"] is not None
        and idx["avg"] is not None
        and profile["ratio_tot_num_avg_low"] is not None
        and profile["ratio_tot_num_avg_high"] is not None
    ):
        denom = x_rows[:, idx["number"]] * x_rows[:, idx["avg"]]
        valid = denom > 1e-9
        ratio = np.divide(
            x_rows[:, idx["tot_sum"]],
            denom,
            out=np.zeros(n_rows, dtype=np.float64),
            where=valid,
        )
        lo = float(profile["ratio_tot_num_avg_low"])
        hi = float(profile["ratio_tot_num_avg_high"])
        v_ratio1 = valid & ((ratio < (lo - tol)) | (ratio > (hi + tol)))

    v_ratio2 = np.zeros(n_rows, dtype=bool)
    if (
        profile["ratio_rate_enabled"]
        and idx["srate"] is not None
        and idx["drate"] is not None
        and idx["rate"] is not None
        and profile["ratio_rate_low"] is not None
        and profile["ratio_rate_high"] is not None
    ):
        rate = x_rows[:, idx["rate"]]
        valid = rate > 1e-9
        numer = x_rows[:, idx["srate"]] + x_rows[:, idx["drate"]]
        ratio = np.divide(numer, rate, out=np.zeros(n_rows, dtype=np.float64), where=valid)
        lo = float(profile["ratio_rate_low"])
        hi = float(profile["ratio_rate_high"])
        v_ratio2 = valid & ((ratio < (lo - tol)) | (ratio > (hi + tol)))

    any_violation = v_min_avg_max | v_ratio1 | v_ratio2
    return {
        "any": any_violation if n_rows > 0 else none_mask,
        "min_avg_max": v_min_avg_max if n_rows > 0 else none_mask,
        "tot_num_avg": v_ratio1 if n_rows > 0 else none_mask,
        "rate_ratio": v_ratio2 if n_rows > 0 else none_mask,
    }


def row_satisfies_profile(
    x_row: np.ndarray,
    x_orig: np.ndarray,
    profile: Dict[str, Any],
    epsilon: float,
    tol: float = 1e-6,
) -> bool:
    lower = profile["lower"]
    upper = profile["upper"]
    if np.any(x_row < (lower - tol)) or np.any(x_row > (upper + tol)):
        return False
    locked_idx = profile["locked_idx"]
    if locked_idx.size > 0 and np.any(np.abs(x_row[locked_idx] - x_orig[locked_idx]) > tol):
        return False
    cap = compute_row_max_delta(x_orig, profile, epsilon)
    if np.any(np.abs(x_row - x_orig) > (cap + tol)):
        return False
    relation_mask = relation_violation_masks(x_row.reshape(1, -1), profile=profile, tol=tol)["any"]
    return not bool(relation_mask[0])


def project_realistic_row(
    x_candidate: np.ndarray,
    x_orig: np.ndarray,
    x_reference_valid: np.ndarray,
    profile: Dict[str, Any],
    epsilon: float,
) -> np.ndarray:
    x = x_candidate.astype(np.float64, copy=True)
    x0 = x_orig.astype(np.float64, copy=False)
    lower = profile["lower"]
    upper = profile["upper"]
    locked_idx = profile["locked_idx"]
    max_delta = compute_row_max_delta(x0, profile, epsilon)

    x = np.clip(x, lower, upper)
    delta = np.clip(x - x0, -max_delta, max_delta)
    x = x0 + delta
    if locked_idx.size > 0:
        x[locked_idx] = x0[locked_idx]

    for _ in range(4):
        prev = x.copy()
        enforce_relation_constraints_row(x, profile)
        x = np.clip(x, lower, upper)
        delta = np.clip(x - x0, -max_delta, max_delta)
        x = x0 + delta
        if locked_idx.size > 0:
            x[locked_idx] = x0[locked_idx]
        if np.max(np.abs(x - prev)) <= 1e-9:
            break

    if row_satisfies_profile(x, x0, profile, epsilon):
        return x.astype(np.float32, copy=False)

    x_ref = x_reference_valid.astype(np.float64, copy=False)
    if row_satisfies_profile(x_ref, x0, profile, epsilon):
        return x_ref.astype(np.float32, copy=True)
    return x0.astype(np.float32, copy=True)


def project_realistic_candidates_fast(
    x_candidates: np.ndarray,
    x_orig: np.ndarray,
    profile: Dict[str, Any],
    epsilon: float,
) -> np.ndarray:
    if x_candidates.size == 0:
        return np.empty((0, x_orig.shape[0]), dtype=np.float32)
    x = x_candidates.astype(np.float64, copy=True)
    x0 = x_orig.astype(np.float64, copy=False)
    lower = profile["lower"]
    upper = profile["upper"]
    locked_idx = profile["locked_idx"]
    max_delta = compute_row_max_delta(x0, profile, epsilon).astype(np.float64, copy=False)

    np.clip(x, lower[None, :], upper[None, :], out=x)
    delta = x - x0[None, :]
    np.clip(delta, -max_delta[None, :], max_delta[None, :], out=delta)
    x = x0[None, :] + delta
    if locked_idx.size > 0:
        x[:, locked_idx] = x0[locked_idx][None, :]
    return x.astype(np.float32, copy=False)


def summarize_realistic_violations(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    profile: Dict[str, Any],
    epsilon: float,
) -> Dict[str, int]:
    if x_adv.size == 0:
        return {
            "below_lower_violations": 0,
            "above_upper_violations": 0,
            "locked_change_rows": 0,
            "locked_change_max_abs": 0,
            "cap_violation_rows": 0,
            "relation_violation_rows": 0,
            "relation_min_avg_max_rows": 0,
            "relation_tot_num_avg_rows": 0,
            "relation_rate_rows": 0,
        }

    x1 = x_adv.astype(np.float64, copy=False)
    x0 = x_orig.astype(np.float64, copy=False)
    lower = profile["lower"]
    upper = profile["upper"]
    locked_idx = profile["locked_idx"]
    unlocked_mask = np.ones(x1.shape[1], dtype=bool)
    if locked_idx.size > 0:
        unlocked_mask[locked_idx] = False

    below = int(np.sum(x1[:, unlocked_mask] < (lower[None, unlocked_mask] - 1e-6)))
    above = int(np.sum(x1[:, unlocked_mask] > (upper[None, unlocked_mask] + 1e-6)))
    if locked_idx.size > 0:
        lock_delta = np.abs(x1[:, locked_idx] - x0[:, locked_idx])
        lock_rows = int(np.sum(np.max(lock_delta, axis=1) > 1e-9))
        lock_max = float(np.max(lock_delta)) if lock_delta.size > 0 else 0.0
    else:
        lock_rows = 0
        lock_max = 0.0

    cap = np.minimum(
        max(0.0, float(epsilon)) * profile["span"][None, :],
        profile["relative_caps"][None, :] * np.maximum(np.abs(x0), 1.0),
    )
    if locked_idx.size > 0:
        cap[:, locked_idx] = 0.0
    cap_violation_rows = int(np.sum(np.max(np.abs(x1 - x0) - cap, axis=1) > 1e-2))

    relation_masks = relation_violation_masks(x1, profile=profile, tol=1e-6)
    return {
        "below_lower_violations": below,
        "above_upper_violations": above,
        "locked_change_rows": lock_rows,
        "locked_change_max_abs": lock_max,
        "cap_violation_rows": cap_violation_rows,
        "relation_violation_rows": int(np.sum(relation_masks["any"])),
        "relation_min_avg_max_rows": int(np.sum(relation_masks["min_avg_max"])),
        "relation_tot_num_avg_rows": int(np.sum(relation_masks["tot_num_avg"])),
        "relation_rate_rows": int(np.sum(relation_masks["rate_ratio"])),
    }


def score_margin_batch(booster: xgb.Booster, x_batch: np.ndarray, threshold: float) -> np.ndarray:
    if x_batch.size == 0:
        return np.empty(0, dtype=np.float64)
    x32 = x_batch.astype(np.float32, copy=False)
    try:
        score = booster.inplace_predict(x32, validate_features=False)
        score_arr = np.asarray(score, dtype=np.float64)
    except Exception:
        score_arr = booster.predict(xgb.DMatrix(x32)).astype(np.float64, copy=False)
    return score_arr - float(threshold)


def propose_sparse_candidates(
    x_current: np.ndarray,
    max_delta_row: np.ndarray,
    mutable_idx: np.ndarray,
    n_candidates: int,
    subset_size: int,
    objective: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_features = x_current.shape[0]
    out = np.repeat(x_current.reshape(1, -1), int(n_candidates), axis=0).astype(np.float64, copy=False)
    if mutable_idx.size == 0 or n_candidates <= 0:
        return out

    max_subset = max(1, min(int(subset_size), int(mutable_idx.size)))
    for i in range(int(n_candidates)):
        k = int(rng.integers(1, max_subset + 1))
        chosen = rng.choice(mutable_idx, size=k, replace=False)
        for j_raw in chosen:
            j = int(j_raw)
            cap = float(max_delta_row[j])
            if cap <= 1e-12:
                continue
            step = float(rng.uniform(0.35, 1.0)) * cap
            if objective == "minimize":
                sign = -1.0 if float(rng.random()) < 0.65 else 1.0
            elif objective == "maximize":
                sign = 1.0 if float(rng.random()) < 0.65 else -1.0
            else:
                sign = -1.0 if float(rng.random()) < 0.5 else 1.0
            out[i, j] = out[i, j] + sign * step
    return out


def run_query_sparse_hillclimb(
    x_orig: np.ndarray,
    baseline_margin: np.ndarray,
    booster: Optional[xgb.Booster],
    threshold: float,
    profile: Dict[str, Any],
    epsilon: float,
    objective: str,
    query_budget: int,
    max_steps: int,
    candidates_per_step: int,
    feature_subset_size: int,
    rng: np.random.Generator,
    progress_prefix: str,
    start_ts: float,
    score_margin_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    score_batch_rows: int = 64,
    max_active_rows_per_step: Optional[int] = None,
    stagnation_patience: Optional[int] = None,
    stagnation_min_delta: float = 1e-6,
    fast_projection: bool = True,
    refine_topk: int = 2,
) -> Dict[str, Any]:
    n_rows = int(x_orig.shape[0])
    x_orig_f64 = x_orig.astype(np.float64, copy=False)
    x_cur = x_orig_f64.copy()
    x_adv = x_orig.astype(np.float32, copy=True)
    queries_used = np.zeros(n_rows, dtype=np.int32)
    accepted_moves = np.zeros(n_rows, dtype=np.int32)
    final_margin = baseline_margin.astype(np.float64, copy=True)

    n_budget = max(0, int(query_budget))
    n_steps = max(1, int(max_steps))
    n_candidates = max(1, int(candidates_per_step))
    subset_size = max(1, int(feature_subset_size))
    mutable_idx = profile["mutable_idx"]
    locked_idx = profile["locked_idx"]
    span = profile["span"].astype(np.float64, copy=False)
    relative_caps = profile["relative_caps"].astype(np.float64, copy=False)
    row_batch_size = max(1, int(score_batch_rows))
    active_cap = int(max_active_rows_per_step) if max_active_rows_per_step is not None else 0
    if active_cap <= 0:
        active_cap = 0
    stagnation_pat = int(stagnation_patience) if stagnation_patience is not None else 0
    if stagnation_pat <= 0:
        stagnation_pat = 0
    min_delta = max(0.0, float(stagnation_min_delta))
    use_fast_projection = bool(fast_projection)
    refine_k = max(1, int(refine_topk))
    tol = 1e-9
    if score_margin_fn is None and booster is None:
        raise ValueError("run_query_sparse_hillclimb requires either booster or score_margin_fn.")

    def _score_margin_local(x_batch: np.ndarray) -> np.ndarray:
        if score_margin_fn is not None:
            return np.asarray(score_margin_fn(x_batch.astype(np.float32, copy=False)), dtype=np.float64)
        assert booster is not None
        return score_margin_batch(booster, x_batch, threshold=threshold)

    if objective == "minimize":
        success = final_margin < 0.0
    else:
        success = final_margin >= 0.0

    if n_rows == 0:
        return {
            "x_adv": x_adv,
            "queries_used": queries_used,
            "accepted_moves": accepted_moves,
            "success": success,
            "final_margin": final_margin,
            "summary": {
                "samples": 0,
                "successes": 0,
                "success_rate": float("nan"),
                "queries_total": 0,
                "queries_mean": float("nan"),
                "queries_p95": float("nan"),
                "accepted_moves_mean": float("nan"),
                "steps_executed": 0,
                "steps_with_progress": 0,
                "active_rows_total": 0,
                "active_rows_mean_per_step": float("nan"),
                "active_rows_p95_per_step": float("nan"),
                "active_rows_max_per_step": 0,
                "max_active_rows_per_step": int(active_cap),
                "stagnation_patience": int(stagnation_pat),
                "stagnation_min_delta": float(min_delta),
                "stagnation_steps": 0,
                "stagnation_triggered": False,
                "fast_projection": bool(use_fast_projection),
                "refine_topk": int(refine_k),
                "fast_projection_rows": 0,
                "full_projection_rows": 0,
                "fast_refine_candidates_scored": 0,
                "full_projection_candidates": 0,
            },
        }

    row_max_delta = np.minimum(
        max(0.0, float(epsilon)) * span.reshape(1, -1),
        relative_caps.reshape(1, -1) * np.maximum(np.abs(x_orig_f64), 1.0),
    ).astype(np.float64, copy=False)
    if locked_idx.size > 0:
        row_max_delta[:, locked_idx] = 0.0

    progress_every = max(50, min(500, n_rows // 10 if n_rows >= 10 else 50))
    loop_start = time.time()
    done_prev = -1
    steps_executed = 0
    steps_with_progress = 0
    stagnation_steps = 0
    stagnation_triggered = False
    fast_projection_rows = 0
    full_projection_rows = 0
    fast_refine_candidates_scored = 0
    full_projection_candidates = 0
    active_rows_by_step: List[int] = []
    if epsilon > 0.0 and n_budget > 0 and mutable_idx.size > 0:
        for _ in range(n_steps):
            active_full = np.flatnonzero((~success) & (queries_used < n_budget))
            if active_full.size <= 0:
                break
            steps_executed += 1
            if active_cap > 0 and int(active_full.size) > active_cap:
                active_scores = final_margin[active_full]
                if objective == "minimize":
                    order = np.argsort(active_scores)[::-1]
                else:
                    order = np.argsort(active_scores)
                active = active_full[order[:active_cap]]
            else:
                active = active_full
            active_rows_by_step.append(int(active.size))
            if active.size <= 0:
                continue

            if objective == "minimize":
                step_best_before = float(np.min(final_margin[active]))
            else:
                step_best_before = float(np.max(final_margin[active]))

            for chunk_start in range(0, int(active.size), row_batch_size):
                chunk_rows = active[chunk_start : chunk_start + row_batch_size]
                staged_rows: List[Dict[str, Any]] = []
                stacked_blocks: List[np.ndarray] = []

                for row_i_raw in chunk_rows.tolist():
                    row_i = int(row_i_raw)
                    remaining = int(n_budget - int(queries_used[row_i]))
                    if remaining <= 0:
                        continue
                    reserve_refine = 0
                    if use_fast_projection and remaining > 2 and int(n_candidates) > 1:
                        reserve_refine = min(int(refine_k), remaining - 1)
                    if reserve_refine > 0:
                        k_candidates = min(int(n_candidates), max(1, remaining - reserve_refine))
                    else:
                        k_candidates = min(int(n_candidates), remaining)
                    if k_candidates <= 1:
                        reserve_refine = 0
                        k_candidates = min(int(n_candidates), remaining)
                    if k_candidates <= 0:
                        continue

                    proposals = propose_sparse_candidates(
                        x_current=x_cur[row_i],
                        max_delta_row=row_max_delta[row_i],
                        mutable_idx=mutable_idx,
                        n_candidates=k_candidates,
                        subset_size=subset_size,
                        objective=objective,
                        rng=rng,
                    )
                    x0 = x_orig_f64[row_i]
                    x_ref = x_cur[row_i].copy()
                    if reserve_refine > 0:
                        proposals_fast = project_realistic_candidates_fast(
                            x_candidates=proposals,
                            x_orig=x0,
                            profile=profile,
                            epsilon=float(epsilon),
                        ).astype(np.float32, copy=False)
                        fast_projection_rows += 1
                        staged_rows.append(
                            {
                                "row_i": int(row_i),
                                "proposals": proposals_fast,
                                "x0": x0,
                                "x_ref": x_ref,
                                "refine_topk": int(reserve_refine),
                            }
                        )
                        stacked_blocks.append(proposals_fast)
                    else:
                        for c in range(k_candidates):
                            proposals[c] = project_realistic_row(
                                x_candidate=proposals[c],
                                x_orig=x0,
                                x_reference_valid=x_ref,
                                profile=profile,
                                epsilon=float(epsilon),
                            )
                        proposals_full = proposals.astype(np.float32, copy=False)
                        full_projection_rows += 1
                        full_projection_candidates += int(k_candidates)
                        staged_rows.append(
                            {
                                "row_i": int(row_i),
                                "proposals": proposals_full,
                                "x0": x0,
                                "x_ref": x_ref,
                                "refine_topk": 0,
                            }
                        )
                        stacked_blocks.append(proposals_full)

                if not staged_rows:
                    continue

                stacked = np.vstack(stacked_blocks).astype(np.float32, copy=False)
                margins_all = _score_margin_local(stacked)
                offset = 0
                for staged in staged_rows:
                    row_i = int(staged["row_i"])
                    proposals = np.asarray(staged["proposals"], dtype=np.float32)
                    k_candidates = int(proposals.shape[0])
                    row_margins = margins_all[offset : offset + k_candidates]
                    offset += k_candidates
                    used_queries = int(k_candidates)

                    refine_cnt = int(staged.get("refine_topk", 0))
                    if refine_cnt > 0 and k_candidates > 0:
                        refine_cnt = int(min(refine_cnt, k_candidates))
                        if objective == "minimize":
                            if refine_cnt >= k_candidates:
                                top_idx = np.arange(k_candidates, dtype=np.int64)
                            else:
                                top_idx = np.argpartition(row_margins, refine_cnt - 1)[:refine_cnt]
                            top_idx = top_idx[np.argsort(row_margins[top_idx])]
                        else:
                            if refine_cnt >= k_candidates:
                                top_idx = np.arange(k_candidates, dtype=np.int64)
                            else:
                                top_idx = np.argpartition(row_margins, -refine_cnt)[-refine_cnt:]
                            top_idx = top_idx[np.argsort(row_margins[top_idx])[::-1]]

                        refined_blocks: List[np.ndarray] = []
                        x0 = np.asarray(staged["x0"], dtype=np.float64)
                        x_ref = np.asarray(staged["x_ref"], dtype=np.float64)
                        for idx_local in top_idx.tolist():
                            refined_blocks.append(
                                project_realistic_row(
                                    x_candidate=proposals[int(idx_local)],
                                    x_orig=x0,
                                    x_reference_valid=x_ref,
                                    profile=profile,
                                    epsilon=float(epsilon),
                                )
                            )
                        refined_stack = np.vstack(refined_blocks).astype(np.float32, copy=False)
                        refined_margins = _score_margin_local(refined_stack)
                        fast_refine_candidates_scored += int(refined_stack.shape[0])
                        used_queries += int(refined_stack.shape[0])
                        proposals_eval = refined_stack
                        row_margins_eval = np.asarray(refined_margins, dtype=np.float64)
                    else:
                        proposals_eval = proposals
                        row_margins_eval = np.asarray(row_margins, dtype=np.float64)

                    queries_used[row_i] = int(queries_used[row_i] + used_queries)

                    margin_cur = float(final_margin[row_i])
                    if objective == "minimize":
                        best_idx = int(np.argmin(row_margins_eval))
                        best_margin = float(row_margins_eval[best_idx])
                        improved = bool(best_margin < (margin_cur - tol))
                    else:
                        best_idx = int(np.argmax(row_margins_eval))
                        best_margin = float(row_margins_eval[best_idx])
                        improved = bool(best_margin > (margin_cur + tol))

                    if improved:
                        x_cur[row_i] = proposals_eval[best_idx].astype(np.float64, copy=False)
                        final_margin[row_i] = float(best_margin)
                        accepted_moves[row_i] = int(accepted_moves[row_i] + 1)
                        if objective == "minimize":
                            success[row_i] = bool(best_margin < 0.0)
                        else:
                            success[row_i] = bool(best_margin >= 0.0)

            if objective == "minimize":
                step_best_after = float(np.min(final_margin[active]))
                step_improvement = max(0.0, step_best_before - step_best_after)
            else:
                step_best_after = float(np.max(final_margin[active]))
                step_improvement = max(0.0, step_best_after - step_best_before)

            if step_improvement > min_delta:
                steps_with_progress += 1
                stagnation_steps = 0
            else:
                stagnation_steps += 1
                if stagnation_pat > 0 and stagnation_steps >= stagnation_pat:
                    stagnation_triggered = True
                    break

            done_now = int(np.sum(success | (queries_used >= n_budget)))
            if (
                done_now == n_rows
                or done_now - done_prev >= progress_every
                or (done_prev < 0 and done_now > 0)
            ):
                elapsed = time.time() - loop_start
                mean_queries = float(np.mean(queries_used.astype(np.float64)))
                success_rate = float(np.mean(success.astype(np.float64)))
                eta_s = (elapsed / max(1, done_now)) * max(0, n_rows - done_now)
                log_progress(
                    (
                        f"{progress_prefix}: {done_now}/{n_rows} samples, "
                        f"mean_queries={mean_queries:.1f}, success_rate={success_rate:.3f}, ETA~{eta_s/60.0:.1f} min"
                    ),
                    start_ts=start_ts,
                )
                done_prev = done_now
            if stagnation_triggered:
                break

    x_adv[:] = x_cur.astype(np.float32, copy=False)

    done_final = int(np.sum(success | (queries_used >= n_budget)))
    if done_final != done_prev:
        elapsed = time.time() - loop_start
        mean_queries = float(np.mean(queries_used.astype(np.float64)))
        success_rate = float(np.mean(success.astype(np.float64)))
        eta_s = (elapsed / max(1, done_final)) * max(0, n_rows - done_final)
        log_progress(
            (
                f"{progress_prefix}: {done_final}/{n_rows} samples, "
                f"mean_queries={mean_queries:.1f}, success_rate={success_rate:.3f}, ETA~{eta_s/60.0:.1f} min"
            ),
            start_ts=start_ts,
        )

    summary = {
        "samples": int(n_rows),
        "successes": int(np.sum(success)),
        "success_rate": float(np.mean(success.astype(np.float64))) if n_rows > 0 else float("nan"),
        "queries_total": int(np.sum(queries_used)),
        "queries_mean": float(np.mean(queries_used.astype(np.float64))) if n_rows > 0 else float("nan"),
        "queries_p95": float(np.percentile(queries_used.astype(np.float64), 95.0)) if n_rows > 0 else float("nan"),
        "accepted_moves_mean": float(np.mean(accepted_moves.astype(np.float64))) if n_rows > 0 else float("nan"),
        "steps_executed": int(steps_executed),
        "steps_with_progress": int(steps_with_progress),
        "active_rows_total": int(np.sum(active_rows_by_step, dtype=np.int64)) if active_rows_by_step else 0,
        "active_rows_mean_per_step": (
            float(np.mean(np.asarray(active_rows_by_step, dtype=np.float64))) if active_rows_by_step else float("nan")
        ),
        "active_rows_p95_per_step": (
            float(np.percentile(np.asarray(active_rows_by_step, dtype=np.float64), 95.0))
            if active_rows_by_step
            else float("nan")
        ),
        "active_rows_max_per_step": int(max(active_rows_by_step)) if active_rows_by_step else 0,
        "max_active_rows_per_step": int(active_cap),
        "stagnation_patience": int(stagnation_pat),
        "stagnation_min_delta": float(min_delta),
        "stagnation_steps": int(stagnation_steps),
        "stagnation_triggered": bool(stagnation_triggered),
        "fast_projection": bool(use_fast_projection),
        "refine_topk": int(refine_k),
        "fast_projection_rows": int(fast_projection_rows),
        "full_projection_rows": int(full_projection_rows),
        "fast_refine_candidates_scored": int(fast_refine_candidates_scored),
        "full_projection_candidates": int(full_projection_candidates),
    }
    return {
        "x_adv": x_adv,
        "queries_used": queries_used,
        "accepted_moves": accepted_moves,
        "success": success,
        "final_margin": final_margin,
        "summary": summary,
    }


def compute_fp_attack_rates(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    adv_pred: np.ndarray,
    scope_mask: np.ndarray,
) -> Dict[str, float]:
    benign_mask = (y_true == 0) & scope_mask
    benign_total = int(np.sum(benign_mask))
    benign_baseline_clean = benign_mask & (baseline_pred == 0)
    benign_clean_total = int(np.sum(benign_baseline_clean))

    flip_to_attack = int(np.sum(benign_baseline_clean & (adv_pred == 1)))
    attack_pred_total = int(np.sum(benign_mask & (adv_pred == 1)))
    return {
        "fp_attack_rate": float(attack_pred_total / benign_total) if benign_total > 0 else float("nan"),
        "fp_attack_rate_num": int(attack_pred_total),
        "fp_attack_rate_den": int(benign_total),
        "fp_attack_rate_conditional": float(flip_to_attack / benign_clean_total) if benign_clean_total > 0 else float("nan"),
        "fp_attack_rate_conditional_num": int(flip_to_attack),
        "fp_attack_rate_conditional_den": int(benign_clean_total),
    }

def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    start_ts = time.time()
    args = parse_args()
    epsilons = parse_epsilons(args.epsilons)
    attack_methods = parse_attack_methods(args.attack_methods)
    realistic_mode = parse_realistic_mode(args.realistic_mode)
    relation_q_low = validate_unit_quantile(args.relation_quantile_low, "--relation-quantile-low")
    relation_q_high = validate_unit_quantile(args.relation_quantile_high, "--relation-quantile-high")
    if relation_q_high < relation_q_low:
        raise ValueError(
            f"--relation-quantile-high ({relation_q_high}) must be >= --relation-quantile-low ({relation_q_low})"
        )
    validate_nonnegative(args.relative_cap_default, "--relative-cap-default")
    if args.relative_cap_rate_group is not None:
        validate_nonnegative(args.relative_cap_rate_group, "--relative-cap-rate-group")
    if args.relative_cap_size_group is not None:
        validate_nonnegative(args.relative_cap_size_group, "--relative-cap-size-group")
    if args.relative_cap_time_group is not None:
        validate_nonnegative(args.relative_cap_time_group, "--relative-cap-time-group")

    run_dir = Path(args.run_dir) if args.run_dir else discover_latest_hpo_run("reports")
    if run_dir is None:
        raise RuntimeError("No run directory found under reports/. Pass --run-dir explicitly.")
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "xgb_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(run_dir)
    models_by_protocol = load_xgb_models_by_protocol(run_dir)
    thresholds_by_protocol = load_thresholds_by_protocol(run_dir)
    resolved_xgb_device = configure_boosters_device(
        models_by_protocol=models_by_protocol,
        feature_count=len(feature_columns),
        requested_device=args.xgb_device,
        start_ts=start_ts,
    )
    protocols = sorted(models_by_protocol.keys())
    default_protocol = choose_default_protocol(protocols)

    log_progress(f"run_dir={run_dir}", start_ts=start_ts)
    log_progress(f"output_dir={output_dir}", start_ts=start_ts)
    log_progress(f"protocols={protocols}", start_ts=start_ts)
    log_progress(f"attack_methods={attack_methods}", start_ts=start_ts)
    log_progress(f"epsilons={epsilons}", start_ts=start_ts)
    log_progress(f"realistic_mode={'on' if realistic_mode else 'off'}", start_ts=start_ts)
    log_progress(f"xgb_inference_device={resolved_xgb_device}", start_ts=start_ts)

    train_sample_df, train_sample_meta = sample_train_rows_by_protocol(
        train_csv=train_csv,
        feature_columns=feature_columns,
        target_per_protocol=int(args.surrogate_train_per_protocol),
        protocols=protocols,
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )

    log_progress("building constraints", start_ts=start_ts)
    constraints, constraints_df = build_constraints(
        train_sample_df=train_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        percentile_lower=float(args.percentile_lower),
        percentile_upper=float(args.percentile_upper),
    )
    log_progress("constraints ready", start_ts=start_ts)

    realism_profiles: Dict[str, Dict[str, Any]] = {}
    realism_profile_json: Dict[str, Any] = {}
    if realistic_mode:
        log_progress("building realism profiles", start_ts=start_ts)
        realism_profiles, realism_profile_json = build_realism_profiles(
            train_sample_df=train_sample_df,
            feature_columns=feature_columns,
            protocols=protocols,
            constraints=constraints,
            relation_q_low=float(relation_q_low),
            relation_q_high=float(relation_q_high),
            relative_cap_default=float(args.relative_cap_default),
            relative_cap_rate_group=args.relative_cap_rate_group,
            relative_cap_size_group=args.relative_cap_size_group,
            relative_cap_time_group=args.relative_cap_time_group,
        )
        log_progress("realism profiles ready", start_ts=start_ts)

    log_progress("training surrogate models", start_ts=start_ts)
    surrogates, surrogate_meta = train_surrogates_by_protocol(
        train_sample_df=train_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        epochs=int(args.surrogate_epochs),
        lr=float(args.surrogate_lr),
        batch_size=int(args.surrogate_batch_size),
        seed=int(args.seed),
        start_ts=start_ts,
    )
    del train_sample_df
    log_progress("surrogate models ready", start_ts=start_ts)

    eval_df, eval_sample_meta = sample_test_rows_balanced(
        test_csv=test_csv,
        feature_columns=feature_columns,
        sample_attack_per_protocol=int(args.sample_attack_per_protocol),
        sample_benign_per_protocol=int(args.sample_benign_per_protocol),
        protocols=protocols,
        seed=int(args.seed) + 1,
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )

    log_progress("running baseline scoring", start_ts=start_ts)
    x_base = prepare_feature_matrix(eval_df.copy(), feature_columns)
    y_true = pd.to_numeric(eval_df["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
    protocol_hint = eval_df["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)

    baseline_pred_df = score_routed_predictions(
        x_matrix=x_base,
        protocol_hint=protocol_hint,
        feature_columns=feature_columns,
        models_by_protocol=models_by_protocol,
        thresholds_by_protocol=thresholds_by_protocol,
        default_protocol=default_protocol,
    )
    baseline_pred = baseline_pred_df["prediction"].to_numpy(dtype=np.int8, copy=False)
    protocol_used = baseline_pred_df["protocol_used"].astype(str).to_numpy(dtype=object, copy=False)

    baseline_global = classification_metrics(y_true, baseline_pred)
    baseline_by_protocol: Dict[str, Dict[str, float]] = {}
    for proto in protocols:
        mask = protocol_used == proto
        if not np.any(mask):
            baseline_by_protocol[proto] = classification_metrics(
                np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int8)
            )
            continue
        baseline_by_protocol[proto] = classification_metrics(y_true[mask], baseline_pred[mask])

    mal_mask_all = y_true == 1
    mal_idx_all = np.flatnonzero(mal_mask_all)
    x_base_mal_all = x_base[mal_idx_all] if mal_idx_all.size > 0 else np.empty((0, len(feature_columns)), dtype=np.float32)
    proto_mal_all = protocol_used[mal_idx_all] if mal_idx_all.size > 0 else np.empty((0,), dtype=object)

    span_by_protocol = {proto: constraints[proto]["span"] for proto in protocols}
    span_mal_all = (
        build_rowwise_tensor(proto_mal_all, span_by_protocol)
        if mal_idx_all.size > 0
        else np.empty((0, len(feature_columns)), dtype=np.float32)
    )

    log_progress("computing SHAP top features per protocol", start_ts=start_ts)
    shap_top_by_protocol: Dict[str, Dict[str, Any]] = {}
    for proto in protocols:
        proto_mal_idx = np.flatnonzero((protocol_used == proto) & (y_true == 1))
        x_proto_mal = x_base[proto_mal_idx] if proto_mal_idx.size > 0 else np.empty((0, len(feature_columns)), dtype=np.float32)
        shap_top_by_protocol[proto] = choose_top_shap_features(
            booster=models_by_protocol[proto],
            x_malicious=x_proto_mal,
            feature_columns=feature_columns,
            locked_mask=constraints[proto]["locked_mask"],
            top_k=int(args.top_shap_features),
        )
        log_progress(
            f"shap[{proto}] top_features={shap_top_by_protocol[proto]['features']}",
            start_ts=start_ts,
        )

    global_rows: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []
    perturb_rows: List[Dict[str, Any]] = []
    by_epsilon_json: Dict[str, Any] = {"baseline": {}, "methods": {}}
    query_global_rows: List[Dict[str, Any]] = []
    query_protocol_rows: List[Dict[str, Any]] = []
    query_trace_summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "query_budget_malicious": int(args.query_budget_malicious),
            "query_budget_benign": int(args.query_budget_benign),
            "query_max_steps": int(args.query_max_steps),
            "query_candidates_per_step": int(args.query_candidates_per_step),
            "query_feature_subset_size": int(args.query_feature_subset_size),
            "query_score_batch_rows": int(args.query_score_batch_rows),
            "query_fast_projection": bool(args.query_fast_projection),
            "query_refine_topk": int(args.query_refine_topk),
            "relation_quantile_low": float(relation_q_low),
            "relation_quantile_high": float(relation_q_high),
            "epsilons": [float(e) for e in epsilons],
        },
        "runs": [],
    }
    realistic_eval_meta: Dict[str, Any] = {}

    base_global_row = {
        "attack_method": "baseline",
        "epsilon": 0.0,
        **baseline_global,
        "asr_conditional": 0.0,
        "asr_conditional_num": 0,
        "asr_conditional_den": int(np.sum((y_true == 1) & (baseline_pred == 1))),
        "asr_all": 0.0,
        "asr_all_num": 0,
        "asr_all_den": int(np.sum(y_true == 1)),
        "delta_f1": 0.0,
    }
    global_rows.append(base_global_row)
    by_epsilon_json["baseline"]["global"] = to_jsonable(base_global_row)

    base_pert_global = summarize_perturbations(
        x_base_mal=x_base_mal_all,
        x_adv_mal=x_base_mal_all,
        span_mal=span_mal_all,
    )
    perturb_rows.append(
        {
            "attack_method": "baseline",
            "epsilon": 0.0,
            "scope": "global",
            "protocol_hint": "",
            **base_pert_global,
            "below_lower_violations": 0,
            "above_upper_violations": 0,
            "locked_change_rows": 0,
            "locked_change_max_abs": 0.0,
        }
    )

    baseline_proto_json: Dict[str, Any] = {}
    for proto in protocols:
        mask = protocol_used == proto
        base_proto_metrics = baseline_by_protocol[proto]
        proto_row = {
            "attack_method": "baseline",
            "epsilon": 0.0,
            "protocol_hint": proto,
            **base_proto_metrics,
            "asr_conditional": 0.0,
            "asr_conditional_num": 0,
            "asr_conditional_den": int(np.sum((y_true == 1) & mask & (baseline_pred == 1))),
            "asr_all": 0.0,
            "asr_all_num": 0,
            "asr_all_den": int(np.sum((y_true == 1) & mask)),
            "delta_f1": 0.0,
        }
        protocol_rows.append(proto_row)
        baseline_proto_json[proto] = to_jsonable(proto_row)

        proto_mal_idx = np.flatnonzero((y_true == 1) & mask)
        x_proto_mal = x_base[proto_mal_idx] if proto_mal_idx.size > 0 else np.empty((0, len(feature_columns)), dtype=np.float32)
        span_proto = np.repeat(constraints[proto]["span"][None, :], x_proto_mal.shape[0], axis=0)
        pert_stats = summarize_perturbations(
            x_base_mal=x_proto_mal,
            x_adv_mal=x_proto_mal,
            span_mal=span_proto,
        )
        perturb_rows.append(
            {
                "attack_method": "baseline",
                "epsilon": 0.0,
                "scope": "protocol",
                "protocol_hint": proto,
                **pert_stats,
                "below_lower_violations": 0,
                "above_upper_violations": 0,
                "locked_change_rows": 0,
                "locked_change_max_abs": 0.0,
            }
        )
    by_epsilon_json["baseline"]["per_protocol"] = baseline_proto_json

    n_total_runs = len(attack_methods) * len(epsilons)
    run_counter = 0

    for method in attack_methods:
        by_epsilon_json["methods"][method] = {}
        for epsilon in epsilons:
            run_counter += 1
            run_start = time.time()
            log_progress(
                f"attack run {run_counter}/{n_total_runs}: method={method}, epsilon={epsilon}",
                start_ts=start_ts,
            )
            x_adv = x_base.copy()
            violation_low = 0
            violation_high = 0
            locked_change_rows = 0
            locked_change_max_abs = 0.0

            for proto in protocols:
                idx = np.flatnonzero((y_true == 1) & (protocol_used == proto))
                if idx.size == 0:
                    continue

                x_orig_proto = x_base[idx]
                lower = constraints[proto]["lower"].astype(np.float64)
                upper = constraints[proto]["upper"].astype(np.float64)
                locked_idx = constraints[proto]["locked_idx"]
                unlocked_mask = np.ones(len(feature_columns), dtype=bool)
                if locked_idx.size > 0:
                    unlocked_mask[locked_idx] = False

                if method == "surrogate_fgsm":
                    x_adv_proto = attack_surrogate_fgsm(
                        x_orig=x_orig_proto,
                        surrogate=surrogates[proto],
                        lower=lower,
                        upper=upper,
                        locked_idx=locked_idx,
                        epsilon=float(epsilon),
                    )
                elif method == "surrogate_pgd":
                    x_adv_proto = attack_surrogate_pgd(
                        x_orig=x_orig_proto,
                        surrogate=surrogates[proto],
                        lower=lower,
                        upper=upper,
                        locked_idx=locked_idx,
                        epsilon=float(epsilon),
                        steps=int(args.pgd_steps),
                        alpha_ratio=float(args.pgd_alpha_ratio),
                    )
                elif method == "heuristic_shap":
                    x_adv_proto = attack_shap_heuristic(
                        x_orig=x_orig_proto,
                        lower=lower,
                        upper=upper,
                        locked_idx=locked_idx,
                        top_indices=shap_top_by_protocol[proto]["indices"],
                        epsilon=float(epsilon),
                    )
                else:
                    raise RuntimeError(f"Unsupported method: {method}")

                x_adv[idx] = x_adv_proto
                below = np.sum(x_adv_proto[:, unlocked_mask] < (lower[None, unlocked_mask] - 1e-6))
                above = np.sum(x_adv_proto[:, unlocked_mask] > (upper[None, unlocked_mask] + 1e-6))
                violation_low += int(below)
                violation_high += int(above)

                if locked_idx.size > 0:
                    lock_delta = np.abs(x_adv_proto[:, locked_idx] - x_orig_proto[:, locked_idx])
                    lock_rows_changed = np.sum(np.max(lock_delta, axis=1) > 1e-9)
                    locked_change_rows += int(lock_rows_changed)
                    if lock_delta.size > 0:
                        locked_change_max_abs = max(locked_change_max_abs, float(np.max(lock_delta)))

            adv_pred_df = score_routed_predictions(
                x_matrix=x_adv,
                protocol_hint=protocol_hint,
                feature_columns=feature_columns,
                models_by_protocol=models_by_protocol,
                thresholds_by_protocol=thresholds_by_protocol,
                default_protocol=default_protocol,
            )
            adv_pred = adv_pred_df["prediction"].to_numpy(dtype=np.int8, copy=False)

            global_metrics = classification_metrics(y_true, adv_pred)
            global_asr = compute_asr(
                y_true=y_true,
                baseline_pred=baseline_pred,
                adv_pred=adv_pred,
                scope_mask=np.ones(len(y_true), dtype=bool),
            )
            global_row = {
                "attack_method": method,
                "epsilon": float(epsilon),
                **global_metrics,
                **global_asr,
                "delta_f1": float(global_metrics["f1"] - baseline_global["f1"]),
            }
            global_rows.append(global_row)

            method_json_entry = {"global": to_jsonable(global_row), "per_protocol": {}, "perturbation": {}}

            for proto in protocols:
                scope_mask = protocol_used == proto
                if not np.any(scope_mask):
                    row = {
                        "attack_method": method,
                        "epsilon": float(epsilon),
                        "protocol_hint": proto,
                        **classification_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int8)),
                        **compute_asr(
                            y_true=np.empty(0, dtype=np.int8),
                            baseline_pred=np.empty(0, dtype=np.int8),
                            adv_pred=np.empty(0, dtype=np.int8),
                            scope_mask=np.empty(0, dtype=bool),
                        ),
                        "delta_f1": float("nan"),
                    }
                else:
                    proto_metrics = classification_metrics(y_true[scope_mask], adv_pred[scope_mask])
                    proto_asr = compute_asr(
                        y_true=y_true,
                        baseline_pred=baseline_pred,
                        adv_pred=adv_pred,
                        scope_mask=scope_mask,
                    )
                    row = {
                        "attack_method": method,
                        "epsilon": float(epsilon),
                        "protocol_hint": proto,
                        **proto_metrics,
                        **proto_asr,
                        "delta_f1": float(proto_metrics["f1"] - baseline_by_protocol[proto]["f1"]),
                    }
                protocol_rows.append(row)
                method_json_entry["per_protocol"][proto] = to_jsonable(row)

            x_adv_mal_all = x_adv[mal_idx_all] if mal_idx_all.size > 0 else np.empty((0, len(feature_columns)), dtype=np.float32)
            pert_global = summarize_perturbations(
                x_base_mal=x_base_mal_all,
                x_adv_mal=x_adv_mal_all,
                span_mal=span_mal_all,
            )
            pert_global_row = {
                "attack_method": method,
                "epsilon": float(epsilon),
                "scope": "global",
                "protocol_hint": "",
                **pert_global,
                "below_lower_violations": int(violation_low),
                "above_upper_violations": int(violation_high),
                "locked_change_rows": int(locked_change_rows),
                "locked_change_max_abs": float(locked_change_max_abs),
            }
            perturb_rows.append(pert_global_row)
            method_json_entry["perturbation"]["global"] = to_jsonable(pert_global_row)

            for proto in protocols:
                mal_scope_idx = np.flatnonzero((y_true == 1) & (protocol_used == proto))
                if mal_scope_idx.size == 0:
                    proto_pert = summarize_perturbations(
                        x_base_mal=np.empty((0, len(feature_columns)), dtype=np.float32),
                        x_adv_mal=np.empty((0, len(feature_columns)), dtype=np.float32),
                        span_mal=np.empty((0, len(feature_columns)), dtype=np.float32),
                    )
                    pert_row = {
                        "attack_method": method,
                        "epsilon": float(epsilon),
                        "scope": "protocol",
                        "protocol_hint": proto,
                        **proto_pert,
                        "below_lower_violations": 0,
                        "above_upper_violations": 0,
                        "locked_change_rows": 0,
                        "locked_change_max_abs": 0.0,
                    }
                else:
                    x0p = x_base[mal_scope_idx]
                    x1p = x_adv[mal_scope_idx]
                    span = np.repeat(constraints[proto]["span"][None, :], x0p.shape[0], axis=0)
                    lower = constraints[proto]["lower"][None, :]
                    upper = constraints[proto]["upper"][None, :]
                    locked_idx = constraints[proto]["locked_idx"]
                    unlocked_mask = np.ones(len(feature_columns), dtype=bool)
                    if locked_idx.size > 0:
                        unlocked_mask[locked_idx] = False

                    proto_pert = summarize_perturbations(
                        x_base_mal=x0p,
                        x_adv_mal=x1p,
                        span_mal=span,
                    )
                    p_below = int(np.sum(x1p[:, unlocked_mask] < (lower[:, unlocked_mask] - 1e-6)))
                    p_above = int(np.sum(x1p[:, unlocked_mask] > (upper[:, unlocked_mask] + 1e-6)))
                    if locked_idx.size > 0:
                        lock_delta = np.abs(x1p[:, locked_idx] - x0p[:, locked_idx])
                        p_locked_rows = int(np.sum(np.max(lock_delta, axis=1) > 1e-9))
                        p_locked_max = float(np.max(lock_delta)) if lock_delta.size > 0 else 0.0
                    else:
                        p_locked_rows = 0
                        p_locked_max = 0.0
                    pert_row = {
                        "attack_method": method,
                        "epsilon": float(epsilon),
                        "scope": "protocol",
                        "protocol_hint": proto,
                        **proto_pert,
                        "below_lower_violations": p_below,
                        "above_upper_violations": p_above,
                        "locked_change_rows": p_locked_rows,
                        "locked_change_max_abs": p_locked_max,
                    }
                perturb_rows.append(pert_row)
                method_json_entry["perturbation"][proto] = to_jsonable(pert_row)

            by_epsilon_json["methods"][method][str(epsilon)] = method_json_entry
            run_elapsed = time.time() - run_start
            remaining = n_total_runs - run_counter
            eta_s = remaining * run_elapsed
            log_progress(
                f"completed method={method}, epsilon={epsilon} in {run_elapsed:.1f}s; ETA~{eta_s/60.0:.1f} min",
                start_ts=start_ts,
            )

    if realistic_mode:
        log_progress("sampling realistic evaluation subset", start_ts=start_ts)
        realistic_eval_df, realistic_eval_meta = sample_test_rows_balanced(
            test_csv=test_csv,
            feature_columns=feature_columns,
            sample_attack_per_protocol=int(args.realistic_sample_attack_per_protocol),
            sample_benign_per_protocol=int(args.realistic_sample_benign_per_protocol),
            protocols=protocols,
            seed=int(args.seed) + 101,
            chunk_size=int(args.chunk_size),
            start_ts=start_ts,
        )

        log_progress("running realistic baseline scoring", start_ts=start_ts)
        x_real_base = prepare_feature_matrix(realistic_eval_df.copy(), feature_columns)
        y_real_true = pd.to_numeric(realistic_eval_df["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
        real_protocol_hint = (
            realistic_eval_df["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
        )
        real_base_pred_df = score_routed_predictions(
            x_matrix=x_real_base,
            protocol_hint=real_protocol_hint,
            feature_columns=feature_columns,
            models_by_protocol=models_by_protocol,
            thresholds_by_protocol=thresholds_by_protocol,
            default_protocol=default_protocol,
        )
        real_base_pred = real_base_pred_df["prediction"].to_numpy(dtype=np.int8, copy=False)
        real_base_score = real_base_pred_df["score_attack"].to_numpy(dtype=np.float64, copy=False)
        real_base_thr = real_base_pred_df["threshold"].to_numpy(dtype=np.float64, copy=False)
        real_protocol_used = real_base_pred_df["protocol_used"].astype(str).to_numpy(dtype=object, copy=False)

        baseline_global_query = classification_metrics(y_real_true, real_base_pred)
        baseline_by_protocol_query: Dict[str, Dict[str, float]] = {}
        for proto in protocols:
            scope = real_protocol_used == proto
            if not np.any(scope):
                baseline_by_protocol_query[proto] = classification_metrics(
                    np.empty(0, dtype=np.int8),
                    np.empty(0, dtype=np.int8),
                )
            else:
                baseline_by_protocol_query[proto] = classification_metrics(y_real_true[scope], real_base_pred[scope])

        base_asr_query = compute_asr(
            y_true=y_real_true,
            baseline_pred=real_base_pred,
            adv_pred=real_base_pred,
            scope_mask=np.ones(len(y_real_true), dtype=bool),
        )
        base_fp_query = compute_fp_attack_rates(
            y_true=y_real_true,
            baseline_pred=real_base_pred,
            adv_pred=real_base_pred,
            scope_mask=np.ones(len(y_real_true), dtype=bool),
        )
        query_global_rows.append(
            {
                "attack_method": "baseline",
                "epsilon": 0.0,
                "attack_objective": "none",
                "target_label": -1,
                "query_budget": 0,
                "targeted_rows": 0,
                "targeted_successes": 0,
                "targeted_success_rate": 0.0,
                "queries_total": 0,
                "queries_mean": 0.0,
                "queries_p95": 0.0,
                "accepted_moves_mean": 0.0,
                **baseline_global_query,
                **base_asr_query,
                **base_fp_query,
                "benign_side_status": "available" if int(base_fp_query["fp_attack_rate_den"]) > 0 else "not_available",
                "delta_f1": 0.0,
                "below_lower_violations": 0,
                "above_upper_violations": 0,
                "locked_change_rows": 0,
                "locked_change_max_abs": 0.0,
                "cap_violation_rows": 0,
                "relation_violation_rows": 0,
                "relation_min_avg_max_rows": 0,
                "relation_tot_num_avg_rows": 0,
                "relation_rate_rows": 0,
            }
        )

        for proto in protocols:
            scope = real_protocol_used == proto
            proto_fp = compute_fp_attack_rates(
                y_true=y_real_true,
                baseline_pred=real_base_pred,
                adv_pred=real_base_pred,
                scope_mask=scope,
            )
            proto_asr = compute_asr(
                y_true=y_real_true,
                baseline_pred=real_base_pred,
                adv_pred=real_base_pred,
                scope_mask=scope,
            )
            query_protocol_rows.append(
                {
                    "attack_method": "baseline",
                    "epsilon": 0.0,
                    "protocol_hint": proto,
                    "attack_objective": "none",
                    "target_label": -1,
                    "query_budget": 0,
                    "targeted_rows": 0,
                    "targeted_successes": 0,
                    "targeted_success_rate": 0.0,
                    "queries_total": 0,
                    "queries_mean": 0.0,
                    "queries_p95": 0.0,
                    "accepted_moves_mean": 0.0,
                    **baseline_by_protocol_query[proto],
                    **proto_asr,
                    **proto_fp,
                    "benign_side_status": "available" if int(proto_fp["fp_attack_rate_den"]) > 0 else "not_available",
                    "delta_f1": 0.0,
                    "below_lower_violations": 0,
                    "above_upper_violations": 0,
                    "locked_change_rows": 0,
                    "locked_change_max_abs": 0.0,
                    "cap_violation_rows": 0,
                    "relation_violation_rows": 0,
                    "relation_min_avg_max_rows": 0,
                    "relation_tot_num_avg_rows": 0,
                    "relation_rate_rows": 0,
                }
            )

        n_query_runs = len(VALID_REALISTIC_METHODS) * len(epsilons)
        q_counter = 0
        for query_method in VALID_REALISTIC_METHODS:
            for epsilon in epsilons:
                q_counter += 1
                run_start = time.time()
                if query_method == "query_sparse_hillclimb":
                    attack_objective = "malicious_evasion"
                    objective = "minimize"
                    target_label = 1
                    query_budget = int(args.query_budget_malicious)
                else:
                    attack_objective = "benign_side_effect"
                    objective = "maximize"
                    target_label = 0
                    query_budget = int(args.query_budget_benign)

                log_progress(
                    (
                        f"realistic run {q_counter}/{n_query_runs}: method={query_method}, "
                        f"epsilon={epsilon}, objective={attack_objective}"
                    ),
                    start_ts=start_ts,
                )
                x_adv_query = x_real_base.copy()
                per_proto_trace: Dict[str, Any] = {}
                all_q: List[np.ndarray] = []
                all_accept: List[np.ndarray] = []
                all_success: List[np.ndarray] = []
                viol_totals = {
                    "below_lower_violations": 0,
                    "above_upper_violations": 0,
                    "locked_change_rows": 0,
                    "locked_change_max_abs": 0.0,
                    "cap_violation_rows": 0,
                    "relation_violation_rows": 0,
                    "relation_min_avg_max_rows": 0,
                    "relation_tot_num_avg_rows": 0,
                    "relation_rate_rows": 0,
                }

                for proto_i, proto in enumerate(protocols):
                    target_idx = np.flatnonzero((real_protocol_used == proto) & (y_real_true == target_label))
                    if target_idx.size == 0:
                        per_proto_trace[proto] = {
                            "targeted_rows": 0,
                            "targeted_successes": 0,
                            "targeted_success_rate": float("nan"),
                            "queries_total": 0,
                            "queries_mean": float("nan"),
                            "queries_p95": float("nan"),
                            "accepted_moves_mean": float("nan"),
                            "below_lower_violations": 0,
                            "above_upper_violations": 0,
                            "locked_change_rows": 0,
                            "locked_change_max_abs": 0.0,
                            "cap_violation_rows": 0,
                            "relation_violation_rows": 0,
                            "relation_min_avg_max_rows": 0,
                            "relation_tot_num_avg_rows": 0,
                            "relation_rate_rows": 0,
                        }
                        continue

                    seed_offset = int(round(float(epsilon) * 1_000_000.0))
                    rng = np.random.default_rng(
                        int(args.seed) + 700_001 + proto_i * 101 + seed_offset * 7 + (0 if target_label == 1 else 1_000_000)
                    )
                    baseline_margin = real_base_score[target_idx] - real_base_thr[target_idx]
                    attack_res = run_query_sparse_hillclimb(
                        x_orig=x_real_base[target_idx],
                        baseline_margin=baseline_margin,
                        booster=models_by_protocol[proto],
                        threshold=float(thresholds_by_protocol.get(proto, 0.5)),
                        profile=realism_profiles[proto],
                        epsilon=float(epsilon),
                        objective=objective,
                        query_budget=int(query_budget),
                        max_steps=int(args.query_max_steps),
                        candidates_per_step=int(args.query_candidates_per_step),
                        feature_subset_size=int(args.query_feature_subset_size),
                        score_batch_rows=int(args.query_score_batch_rows),
                        max_active_rows_per_step=(
                            int(args.query_max_active_rows_per_step)
                            if int(args.query_max_active_rows_per_step) > 0
                            else None
                        ),
                        stagnation_patience=(
                            int(args.query_stagnation_patience)
                            if int(args.query_stagnation_patience) > 0
                            else None
                        ),
                        stagnation_min_delta=float(args.query_stagnation_min_delta),
                        fast_projection=bool(args.query_fast_projection),
                        refine_topk=max(1, int(args.query_refine_topk)),
                        rng=rng,
                        progress_prefix=f"{query_method}[{proto}] eps={epsilon}",
                        start_ts=start_ts,
                    )

                    x_adv_proto = attack_res["x_adv"]
                    x_adv_query[target_idx] = x_adv_proto
                    all_q.append(attack_res["queries_used"])
                    all_accept.append(attack_res["accepted_moves"])
                    all_success.append(attack_res["success"].astype(np.int8, copy=False))

                    changed_mask = (
                        np.max(np.abs(x_adv_proto.astype(np.float64) - x_real_base[target_idx].astype(np.float64)), axis=1) > 1e-9
                    )
                    if np.any(changed_mask):
                        viol = summarize_realistic_violations(
                            x_adv=x_adv_proto[changed_mask],
                            x_orig=x_real_base[target_idx][changed_mask],
                            profile=realism_profiles[proto],
                            epsilon=float(epsilon),
                        )
                    else:
                        viol = {
                            "below_lower_violations": 0,
                            "above_upper_violations": 0,
                            "locked_change_rows": 0,
                            "locked_change_max_abs": 0.0,
                            "cap_violation_rows": 0,
                            "relation_violation_rows": 0,
                            "relation_min_avg_max_rows": 0,
                            "relation_tot_num_avg_rows": 0,
                            "relation_rate_rows": 0,
                        }
                    for key, value in viol.items():
                        if key == "locked_change_max_abs":
                            viol_totals[key] = float(max(float(viol_totals[key]), float(value)))
                        else:
                            viol_totals[key] += int(value)

                    proto_summary = attack_res["summary"]
                    per_proto_trace[proto] = {
                        "targeted_rows": int(target_idx.size),
                        "targeted_successes": int(proto_summary["successes"]),
                        "targeted_success_rate": float(proto_summary["success_rate"]),
                        "queries_total": int(proto_summary["queries_total"]),
                        "queries_mean": float(proto_summary["queries_mean"]),
                        "queries_p95": float(proto_summary["queries_p95"]),
                        "accepted_moves_mean": float(proto_summary["accepted_moves_mean"]),
                        **{k: int(v) if k != "locked_change_max_abs" else float(v) for k, v in viol.items()},
                    }

                if all_q:
                    all_q_concat = np.concatenate(all_q)
                    all_accept_concat = np.concatenate(all_accept)
                    all_success_concat = np.concatenate(all_success).astype(np.float64)
                    targeted_rows = int(all_q_concat.size)
                    targeted_successes = int(np.sum(all_success_concat))
                    targeted_success_rate = float(np.mean(all_success_concat))
                    queries_total = int(np.sum(all_q_concat))
                    queries_mean = float(np.mean(all_q_concat.astype(np.float64)))
                    queries_p95 = float(np.percentile(all_q_concat.astype(np.float64), 95.0))
                    accepted_moves_mean = float(np.mean(all_accept_concat.astype(np.float64)))
                else:
                    targeted_rows = 0
                    targeted_successes = 0
                    targeted_success_rate = float("nan")
                    queries_total = 0
                    queries_mean = float("nan")
                    queries_p95 = float("nan")
                    accepted_moves_mean = float("nan")

                adv_pred_df_query = score_routed_predictions(
                    x_matrix=x_adv_query,
                    protocol_hint=real_protocol_hint,
                    feature_columns=feature_columns,
                    models_by_protocol=models_by_protocol,
                    thresholds_by_protocol=thresholds_by_protocol,
                    default_protocol=default_protocol,
                )
                adv_pred_query = adv_pred_df_query["prediction"].to_numpy(dtype=np.int8, copy=False)

                global_metrics_query = classification_metrics(y_real_true, adv_pred_query)
                global_asr_query = compute_asr(
                    y_true=y_real_true,
                    baseline_pred=real_base_pred,
                    adv_pred=adv_pred_query,
                    scope_mask=np.ones(len(y_real_true), dtype=bool),
                )
                global_fp_query = compute_fp_attack_rates(
                    y_true=y_real_true,
                    baseline_pred=real_base_pred,
                    adv_pred=adv_pred_query,
                    scope_mask=np.ones(len(y_real_true), dtype=bool),
                )
                query_global_rows.append(
                    {
                        "attack_method": query_method,
                        "epsilon": float(epsilon),
                        "attack_objective": attack_objective,
                        "target_label": int(target_label),
                        "query_budget": int(query_budget),
                        "targeted_rows": int(targeted_rows),
                        "targeted_successes": int(targeted_successes),
                        "targeted_success_rate": float(targeted_success_rate),
                        "queries_total": int(queries_total),
                        "queries_mean": float(queries_mean),
                        "queries_p95": float(queries_p95),
                        "accepted_moves_mean": float(accepted_moves_mean),
                        **global_metrics_query,
                        **global_asr_query,
                        **global_fp_query,
                        "benign_side_status": "available" if int(global_fp_query["fp_attack_rate_den"]) > 0 else "not_available",
                        "delta_f1": float(global_metrics_query["f1"] - baseline_global_query["f1"]),
                        **viol_totals,
                    }
                )

                for proto in protocols:
                    scope = real_protocol_used == proto
                    if not np.any(scope):
                        proto_metrics = classification_metrics(
                            np.empty(0, dtype=np.int8),
                            np.empty(0, dtype=np.int8),
                        )
                    else:
                        proto_metrics = classification_metrics(y_real_true[scope], adv_pred_query[scope])
                    proto_asr = compute_asr(
                        y_true=y_real_true,
                        baseline_pred=real_base_pred,
                        adv_pred=adv_pred_query,
                        scope_mask=scope,
                    )
                    proto_fp = compute_fp_attack_rates(
                        y_true=y_real_true,
                        baseline_pred=real_base_pred,
                        adv_pred=adv_pred_query,
                        scope_mask=scope,
                    )
                    trace_info = per_proto_trace.get(
                        proto,
                        {
                            "targeted_rows": 0,
                            "targeted_successes": 0,
                            "targeted_success_rate": float("nan"),
                            "queries_total": 0,
                            "queries_mean": float("nan"),
                            "queries_p95": float("nan"),
                            "accepted_moves_mean": float("nan"),
                            "below_lower_violations": 0,
                            "above_upper_violations": 0,
                            "locked_change_rows": 0,
                            "locked_change_max_abs": 0.0,
                            "cap_violation_rows": 0,
                            "relation_violation_rows": 0,
                            "relation_min_avg_max_rows": 0,
                            "relation_tot_num_avg_rows": 0,
                            "relation_rate_rows": 0,
                        },
                    )
                    query_protocol_rows.append(
                        {
                            "attack_method": query_method,
                            "epsilon": float(epsilon),
                            "protocol_hint": proto,
                            "attack_objective": attack_objective,
                            "target_label": int(target_label),
                            "query_budget": int(query_budget),
                            "targeted_rows": int(trace_info["targeted_rows"]),
                            "targeted_successes": int(trace_info["targeted_successes"]),
                            "targeted_success_rate": float(trace_info["targeted_success_rate"]),
                            "queries_total": int(trace_info["queries_total"]),
                            "queries_mean": float(trace_info["queries_mean"]),
                            "queries_p95": float(trace_info["queries_p95"]),
                            "accepted_moves_mean": float(trace_info["accepted_moves_mean"]),
                            **proto_metrics,
                            **proto_asr,
                            **proto_fp,
                            "benign_side_status": "available" if int(proto_fp["fp_attack_rate_den"]) > 0 else "not_available",
                            "delta_f1": float(proto_metrics["f1"] - baseline_by_protocol_query[proto]["f1"]),
                            "below_lower_violations": int(trace_info["below_lower_violations"]),
                            "above_upper_violations": int(trace_info["above_upper_violations"]),
                            "locked_change_rows": int(trace_info["locked_change_rows"]),
                            "locked_change_max_abs": float(trace_info["locked_change_max_abs"]),
                            "cap_violation_rows": int(trace_info["cap_violation_rows"]),
                            "relation_violation_rows": int(trace_info["relation_violation_rows"]),
                            "relation_min_avg_max_rows": int(trace_info["relation_min_avg_max_rows"]),
                            "relation_tot_num_avg_rows": int(trace_info["relation_tot_num_avg_rows"]),
                            "relation_rate_rows": int(trace_info["relation_rate_rows"]),
                        }
                    )

                query_trace_summary["runs"].append(
                    {
                        "attack_method": query_method,
                        "epsilon": float(epsilon),
                        "attack_objective": attack_objective,
                        "target_label": int(target_label),
                        "query_budget": int(query_budget),
                        "targeted_rows": int(targeted_rows),
                        "targeted_successes": int(targeted_successes),
                        "targeted_success_rate": float(targeted_success_rate),
                        "queries_total": int(queries_total),
                        "queries_mean": float(queries_mean),
                        "queries_p95": float(queries_p95),
                        "accepted_moves_mean": float(accepted_moves_mean),
                        "protocols": to_jsonable(per_proto_trace),
                    }
                )
                run_elapsed = time.time() - run_start
                remaining = n_query_runs - q_counter
                eta_s = run_elapsed * remaining
                log_progress(
                    (
                        f"completed realistic method={query_method}, epsilon={epsilon} in {run_elapsed:.1f}s; "
                        f"ETA~{eta_s/60.0:.1f} min"
                    ),
                    start_ts=start_ts,
                )

    global_df = pd.DataFrame(global_rows)
    protocol_df = pd.DataFrame(protocol_rows)
    perturb_df = pd.DataFrame(perturb_rows)
    constraints_df = constraints_df.sort_values(["protocol_hint", "feature"]).reset_index(drop=True)

    global_csv = output_dir / "robustness_metrics_global.csv"
    protocol_csv = output_dir / "robustness_metrics_protocol.csv"
    perturb_csv = output_dir / "perturbation_stats.csv"
    constraints_csv = output_dir / "constraints_summary.csv"
    by_eps_json_path = output_dir / "robustness_metrics_by_epsilon.json"
    attack_cfg_json_path = output_dir / "attack_config.json"
    summary_json_path = output_dir / "summary.json"
    query_global_csv = output_dir / "robustness_query_metrics_global.csv"
    query_protocol_csv = output_dir / "robustness_query_metrics_protocol.csv"
    realism_profile_json_path = output_dir / "realism_profile.json"
    query_trace_json_path = output_dir / "query_trace_summary.json"

    global_df.to_csv(global_csv, index=False)
    protocol_df.to_csv(protocol_csv, index=False)
    perturb_df.to_csv(perturb_csv, index=False)
    constraints_df.to_csv(constraints_csv, index=False)
    if realistic_mode:
        pd.DataFrame(query_global_rows).to_csv(query_global_csv, index=False)
        pd.DataFrame(query_protocol_rows).to_csv(query_protocol_csv, index=False)
        with realism_profile_json_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(realism_profile_json), f, indent=2)
        with query_trace_json_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(query_trace_summary), f, indent=2)

    with by_eps_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(by_epsilon_json), f, indent=2)

    attack_config = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "protocols": protocols,
        "default_protocol": default_protocol,
        "thresholds_by_protocol": {k: float(v) for k, v in thresholds_by_protocol.items()},
        "xgb_device_requested": str(args.xgb_device),
        "xgb_device_resolved": str(resolved_xgb_device),
        "attack_methods": attack_methods,
        "epsilons": [float(e) for e in epsilons],
        "sample_attack_per_protocol": int(args.sample_attack_per_protocol),
        "sample_benign_per_protocol": int(args.sample_benign_per_protocol),
        "surrogate_train_per_protocol": int(args.surrogate_train_per_protocol),
        "surrogate_epochs": int(args.surrogate_epochs),
        "surrogate_lr": float(args.surrogate_lr),
        "surrogate_batch_size": int(args.surrogate_batch_size),
        "pgd_steps": int(args.pgd_steps),
        "pgd_alpha_ratio": float(args.pgd_alpha_ratio),
        "top_shap_features": int(args.top_shap_features),
        "percentile_lower": float(args.percentile_lower),
        "percentile_upper": float(args.percentile_upper),
        "chunk_size": int(args.chunk_size),
        "realistic_mode": "on" if realistic_mode else "off",
        "realistic_methods": list(VALID_REALISTIC_METHODS),
        "query_budget_malicious": int(args.query_budget_malicious),
        "query_budget_benign": int(args.query_budget_benign),
        "query_max_steps": int(args.query_max_steps),
        "query_candidates_per_step": int(args.query_candidates_per_step),
        "query_feature_subset_size": int(args.query_feature_subset_size),
        "relative_cap_default": float(args.relative_cap_default),
        "relative_cap_rate_group": None if args.relative_cap_rate_group is None else float(args.relative_cap_rate_group),
        "relative_cap_size_group": None if args.relative_cap_size_group is None else float(args.relative_cap_size_group),
        "relative_cap_time_group": None if args.relative_cap_time_group is None else float(args.relative_cap_time_group),
        "relation_quantile_low": float(relation_q_low),
        "relation_quantile_high": float(relation_q_high),
        "realistic_sample_attack_per_protocol": int(args.realistic_sample_attack_per_protocol),
        "realistic_sample_benign_per_protocol": int(args.realistic_sample_benign_per_protocol),
    }
    with attack_cfg_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(attack_config), f, indent=2)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "files": {
            "robustness_metrics_global_csv": str(global_csv),
            "robustness_metrics_protocol_csv": str(protocol_csv),
            "robustness_metrics_by_epsilon_json": str(by_eps_json_path),
            "perturbation_stats_csv": str(perturb_csv),
            "constraints_summary_csv": str(constraints_csv),
            "attack_config_json": str(attack_cfg_json_path),
            "robustness_query_metrics_global_csv": str(query_global_csv) if realistic_mode else None,
            "robustness_query_metrics_protocol_csv": str(query_protocol_csv) if realistic_mode else None,
            "realism_profile_json": str(realism_profile_json_path) if realistic_mode else None,
            "query_trace_summary_json": str(query_trace_json_path) if realistic_mode else None,
        },
        "feature_count": int(len(feature_columns)),
        "protocols": protocols,
        "xgb_inference_device": str(resolved_xgb_device),
        "train_sampling": train_sample_meta,
        "eval_sampling": eval_sample_meta,
        "surrogate_meta": surrogate_meta,
        "realistic_eval_sampling": realistic_eval_meta if realistic_mode else None,
        "baseline_global": baseline_global,
        "baseline_by_protocol": baseline_by_protocol,
        "realistic_query_rows_global": int(len(query_global_rows)) if realistic_mode else 0,
        "realistic_query_rows_protocol": int(len(query_protocol_rows)) if realistic_mode else 0,
        "shap_top_features_by_protocol": {
            proto: {
                "features": shap_top_by_protocol[proto]["features"],
                "mean_contrib": shap_top_by_protocol[proto]["mean_contrib"],
            }
            for proto in protocols
        },
        "notes": [
            "Robustness evaluation is inference-only; XGBoost models are not retrained.",
            "Legacy attacks are applied to malicious rows only; benign rows remain unchanged in legacy mode.",
            "Realistic mode adds query-limited black-box attacks for both malicious evasion and benign-side FPR drift.",
            "Constraints use train-sampled percentiles with non-negativity projection and locked feature invariants.",
        ],
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    log_progress(f"saved: {global_csv}", start_ts=start_ts)
    log_progress(f"saved: {protocol_csv}", start_ts=start_ts)
    log_progress(f"saved: {by_eps_json_path}", start_ts=start_ts)
    log_progress(f"saved: {perturb_csv}", start_ts=start_ts)
    log_progress(f"saved: {constraints_csv}", start_ts=start_ts)
    log_progress(f"saved: {attack_cfg_json_path}", start_ts=start_ts)
    log_progress(f"saved: {summary_json_path}", start_ts=start_ts)
    if realistic_mode:
        log_progress(f"saved: {query_global_csv}", start_ts=start_ts)
        log_progress(f"saved: {query_protocol_csv}", start_ts=start_ts)
        log_progress(f"saved: {realism_profile_json_path}", start_ts=start_ts)
        log_progress(f"saved: {query_trace_json_path}", start_ts=start_ts)
    log_progress("robustness evaluation complete", start_ts=start_ts)


if __name__ == "__main__":
    main()
