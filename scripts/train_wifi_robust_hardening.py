
                      

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from evaluate_xgb_robustness import (
    build_constraints,
    build_realism_profiles,
    configure_boosters_device,
    log_progress,
    normalize_label_series,
    run_query_sparse_hillclimb,
    score_margin_batch,
    summarize_realistic_violations,
    to_jsonable,
)
from xgb_protocol_ids_utils import (
    discover_latest_hpo_run,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    protocol_slug,
)


def parse_float_csv(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("Empty float list")
    unique = sorted(set(out))
    for v in unique:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"Invalid epsilon value: {v}")
    return unique


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-run-dir",
        type=str,
        default=None,
        help="Base full_gpu_hpo_models_* run directory (defaults to latest).",
    )
    ap.add_argument("--train-csv", type=str, default="data/merged/metadata_train.csv")
    ap.add_argument("--test-csv", type=str, default="data/merged/metadata_test.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=str, default="reports")
    ap.add_argument("--chunk-size", type=int, default=300000)

    ap.add_argument("--hardneg-train-benign-sample", type=int, default=12000)
    ap.add_argument("--hardneg-val-benign-sample", type=int, default=4000)
    ap.add_argument("--hardneg-epsilons", type=str, default="0.05,0.10")
    ap.add_argument("--hardneg-query-budget", type=int, default=120)
    ap.add_argument("--hardneg-query-max-steps", type=int, default=60)
    ap.add_argument("--hardneg-candidates-per-step", type=int, default=3)
    ap.add_argument("--hardneg-feature-subset-size", type=int, default=3)
    ap.add_argument(
        "--hardneg-extra-topk-per-epsilon",
        type=int,
        default=0,
        help="Per-epsilon extra benign near-miss rows (highest final margin) to include as hard negatives.",
    )
    ap.add_argument("--hardneg-weight", type=float, default=2.0)

    ap.add_argument("--threshold-grid-size", type=int, default=400)
    ap.add_argument("--wifi-recall-target", type=float, default=0.995)
    ap.add_argument(
        "--threshold-adv-malicious-recall-target",
        type=float,
        default=0.995,
        help="Minimum adversarial malicious recall on validation during threshold search (<=0 disables).",
    )
    ap.add_argument(
        "--threshold-fallback-adv-shortfall-weight",
        type=float,
        default=3.0,
        help="Fallback objective weight for adversarial malicious recall shortfall.",
    )
    ap.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument(
        "--val-malicious-sample",
        type=int,
        default=4000,
        help="Validation malicious sample size for adversarial recall calibration.",
    )
    ap.add_argument("--val-malicious-query-budget", type=int, default=None)
    ap.add_argument("--val-malicious-query-max-steps", type=int, default=None)
    ap.add_argument("--val-malicious-candidates-per-step", type=int, default=None)
    ap.add_argument("--val-malicious-feature-subset-size", type=int, default=None)

    ap.add_argument("--val-mod", type=int, default=5)
    ap.add_argument("--percentile-lower", type=float, default=1.0)
    ap.add_argument("--percentile-upper", type=float, default=99.0)
    ap.add_argument("--relation-quantile-low", type=float, default=0.01)
    ap.add_argument("--relation-quantile-high", type=float, default=0.99)
    ap.add_argument("--relative-cap-default", type=float, default=0.20)
    ap.add_argument("--relative-cap-rate-group", type=float, default=None)
    ap.add_argument("--relative-cap-size-group", type=float, default=None)
    ap.add_argument("--relative-cap-time-group", type=float, default=None)

    ap.add_argument("--constraint-sample-size", type=int, default=200000)
    ap.add_argument("--preview-max-rows", type=int, default=2000)
    return ap.parse_args()


def _binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
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
    }


def _select_threshold(
    y_clean: np.ndarray,
    p_clean: np.ndarray,
    p_adv_benign: np.ndarray,
    p_adv_malicious: np.ndarray,
    recall_target: float,
    adv_malicious_recall_target: float,
    adv_shortfall_weight: float,
    grid_size: int,
    base_threshold: float,
) -> Dict[str, Any]:
    y = np.asarray(y_clean, dtype=np.int8)
    p = np.asarray(p_clean, dtype=np.float64)
    p_adv = np.asarray(p_adv_benign, dtype=np.float64)

    all_scores = p[np.isfinite(p)]
    if p_adv.size > 0:
        all_scores = np.concatenate([all_scores, p_adv[np.isfinite(p_adv)]], axis=0)
    if all_scores.size <= 0:
        all_scores = np.array([0.5], dtype=np.float64)

    lo = float(max(0.0, np.min(all_scores)))
    hi = float(min(1.0, np.max(all_scores)))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi < lo:
        hi = lo

    if abs(hi - lo) < 1e-12:
        grid = np.array([lo], dtype=np.float64)
    else:
        grid = np.linspace(lo, hi, num=max(2, int(grid_size)), dtype=np.float64)
    candidates = np.unique(np.concatenate([grid, np.array([float(base_threshold)], dtype=np.float64)], axis=0))

    feasible_rows: List[Dict[str, Any]] = []
    fallback_rows: List[Dict[str, Any]] = []
    has_adv_malicious = p_adv_malicious.size > 0
    for thr in candidates:
        pred = (p >= thr).astype(np.int8, copy=False)
        m = _binary_metrics(y, pred)
        attacked_fpr = float(np.mean((p_adv >= thr).astype(np.float64))) if p_adv.size > 0 else 0.0
        attacked_malicious_recall = (
            float(np.mean((p_adv_malicious >= thr).astype(np.float64))) if has_adv_malicious else float("nan")
        )
        recall_shortfall = max(0.0, float(recall_target) - float(m["recall"]))
        adv_recall_shortfall = (
            max(0.0, float(adv_malicious_recall_target) - attacked_malicious_recall)
            if has_adv_malicious and float(adv_malicious_recall_target) > 0.0
            else 0.0
        )
        row = {
            "threshold": float(thr),
            "clean_f1": float(m["f1"]),
            "clean_recall": float(m["recall"]),
            "clean_precision": float(m["precision"]),
            "clean_fpr": float(m["fpr"]),
            "attacked_benign_fpr": attacked_fpr,
            "attacked_malicious_recall": attacked_malicious_recall,
            "recall_shortfall": recall_shortfall,
            "adv_malicious_recall_shortfall": adv_recall_shortfall,
            "fallback_objective": float(
                m["f1"] - 3.0 * attacked_fpr - 2.0 * recall_shortfall - float(adv_shortfall_weight) * adv_recall_shortfall
            ),
        }
        fallback_rows.append(row)
        clean_ok = float(m["recall"]) >= float(recall_target)
        adv_ok = (
            True
            if (not has_adv_malicious or float(adv_malicious_recall_target) <= 0.0)
            else attacked_malicious_recall >= float(adv_malicious_recall_target)
        )
        if clean_ok and adv_ok:
            feasible_rows.append(row)

    if feasible_rows:
        best = sorted(
            feasible_rows,
            key=lambda r: (
                float(r["attacked_benign_fpr"]),
                float(r["adv_malicious_recall_shortfall"]),
                -float(r["clean_f1"]),
                -float(r["threshold"]),
            ),
        )[0]
        rule = "feasible_min_attacked_benign_fpr_with_clean_and_adv_recall_constraints_then_max_clean_f1"
    else:
        best = max(
            fallback_rows,
            key=lambda r: (float(r["fallback_objective"]), -float(r["attacked_benign_fpr"]), -float(r["threshold"])),
        )
        rule = "fallback_max_clean_f1_minus_3fpr_minus_2clean_shortfall_minus_w_adv_shortfall"

    return {
        "selected_threshold": float(best["threshold"]),
        "selection_rule": rule,
        "selected_metrics": best,
        "num_candidates": int(candidates.size),
        "num_feasible": int(len(feasible_rows)),
        "adv_malicious_constraint_enabled": bool(has_adv_malicious and float(adv_malicious_recall_target) > 0.0),
        "adv_malicious_recall_target": float(adv_malicious_recall_target),
    }

def _build_protocol_class_val_mask(
    df: pd.DataFrame,
    seed: int,
    val_mod: int,
    allow_row_fallback: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if val_mod <= 1:
        raise ValueError("val_mod must be >= 2")
    protocol = df["protocol_hint"].fillna("unknown").astype(str).map(protocol_slug).to_numpy()
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
    rel_hash = pd.util.hash_pandas_object(
        df["source_relpath"].astype("string"),
        index=False,
    ).to_numpy(np.uint64)
    row_idx = pd.to_numeric(df["source_row_index"], errors="coerce").fillna(0).astype(np.uint64).to_numpy(copy=False)

    val_mask = np.zeros(len(df), dtype=bool)
    seed_u64 = np.uint64(seed & 0xFFFFFFFF)
    mod_u64 = np.uint64(val_mod)
    mix_const = np.uint64(11400714819323198485)
    fallback_events: List[Dict[str, Any]] = []

    for proto in sorted(set(protocol.tolist())):
        proto_mask = protocol == proto
        for cls in (0, 1):
            stratum_idx = np.flatnonzero(proto_mask & (y == cls))
            n = int(len(stratum_idx))
            if n <= 1:
                continue

            stratum_rel_hash = rel_hash[stratum_idx]
            unique_files, inv = np.unique(stratum_rel_hash, return_inverse=True)
            n_files = int(len(unique_files))

            file_val_primary = ((unique_files + seed_u64) % mod_u64) == 0
            stratum_val = file_val_primary[inv]
            fallback_type: Optional[str] = None

            if not stratum_val.any() or stratum_val.all():
                if n_files >= 2:
                    with np.errstate(over="ignore"):
                        mixed_files = unique_files ^ (seed_u64 * mix_const)
                    file_order = np.argsort(mixed_files, kind="mergesort")
                    n_val_files = max(1, min(n_files - 1, int(round(n_files / float(val_mod)))))
                    file_val_rebalanced = np.zeros(n_files, dtype=bool)
                    file_val_rebalanced[file_order[:n_val_files]] = True
                    stratum_val = file_val_rebalanced[inv]
                    fallback_type = "file_level_rebalance"
                else:
                    if not allow_row_fallback:
                        raise RuntimeError(
                            "Validation split collapsed for single-file stratum while row fallback is disabled: "
                            f"protocol={proto} class={cls} rows={n}"
                        )
                    with np.errstate(over="ignore"):
                        mixed_rows = stratum_rel_hash ^ (row_idx[stratum_idx] * mix_const) ^ seed_u64
                    row_order = np.argsort(mixed_rows, kind="mergesort")
                    n_val_rows = max(1, min(n - 1, int(round(n / float(val_mod)))))
                    stratum_val = np.zeros(n, dtype=bool)
                    stratum_val[row_order[:n_val_rows]] = True
                    fallback_type = "row_level_single_file"

            if fallback_type is not None:
                fallback_events.append(
                    {
                        "protocol_hint": proto,
                        "label": int(cls),
                        "stratum_rows": int(n),
                        "stratum_files": int(n_files),
                        "fallback_type": fallback_type,
                        "val_rows_after_split": int(np.sum(stratum_val)),
                        "train_rows_after_split": int(n - np.sum(stratum_val)),
                    }
                )
            val_mask[stratum_idx[stratum_val]] = True

    return val_mask, fallback_events


def _load_wifi_train_rows(
    train_csv: Path,
    feature_columns: List[str],
    chunk_size: int,
    start_ts: float,
) -> pd.DataFrame:
    usecols = ["source_relpath", "source_row_index", "protocol_hint", "label"] + feature_columns
    dtypes = {c: "float32" for c in feature_columns}
    dtypes.update(
        {
            "source_relpath": "string",
            "source_row_index": "int32",
            "protocol_hint": "string",
            "label": "int8",
        }
    )

    pieces: List[pd.DataFrame] = []
    total_rows = 0
    wifi_rows = 0
    reader = pd.read_csv(train_csv, usecols=usecols, dtype=dtypes, chunksize=max(1, int(chunk_size)))
    for i, chunk in enumerate(reader, start=1):
        total_rows += int(len(chunk))
        proto = np.array(chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(), dtype=object)
        label = normalize_label_series(chunk["label"])
        mask = proto == "wifi"
        if np.any(mask):
            sub = chunk.loc[mask, ["source_relpath", "source_row_index"] + feature_columns].copy()
            sub["protocol_hint"] = "wifi"
            sub["label"] = label[mask]
            pieces.append(sub)
            wifi_rows += int(len(sub))
        if i % 4 == 0:
            log_progress(
                f"wifi load pass: chunks={i}, scanned_rows={total_rows}, wifi_rows={wifi_rows}",
                start_ts=start_ts,
            )

    if not pieces:
        raise RuntimeError("No WiFi rows found in training CSV.")
    out = pd.concat(pieces, ignore_index=True)
    out["protocol_hint"] = out["protocol_hint"].astype(str).map(protocol_slug)
    out["label"] = normalize_label_series(out["label"])
    return out


def _sample_indices(total: int, want: int, rng: np.random.Generator) -> np.ndarray:
    n = int(total)
    k = int(max(0, min(int(want), n)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    idx = rng.choice(n, size=k, replace=False).astype(np.int64, copy=False)
    idx.sort()
    return idx


def _empty_violation_dict() -> Dict[str, Any]:
    return {
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

def _run_benign_query_campaign(
    *,
    split_name: str,
    x_benign: np.ndarray,
    src_relpath: np.ndarray,
    src_row_index: np.ndarray,
    booster: xgb.Booster,
    threshold: float,
    profile: Dict[str, Any],
    epsilons: Sequence[float],
    query_budget: int,
    query_max_steps: int,
    query_candidates_per_step: int,
    query_feature_subset_size: int,
    extra_topk_per_epsilon: int,
    seed: int,
    start_ts: float,
) -> Dict[str, Any]:
    n_rows, n_features = x_benign.shape
    if n_rows <= 0:
        return {
            "flipped_adv": np.empty((0, n_features), dtype=np.float32),
            "all_adv": np.empty((0, n_features), dtype=np.float32),
            "all_adv_eps": np.empty(0, dtype=np.float64),
            "stats_rows": [],
            "preview_rows": [],
        }

    baseline_margin = score_margin_batch(booster, x_benign.astype(np.float32, copy=False), threshold=float(threshold))
    baseline_benign = baseline_margin < 0.0

    flipped_chunks: List[np.ndarray] = []
    selected_chunks: List[np.ndarray] = []
    all_adv_chunks: List[np.ndarray] = []
    all_eps_chunks: List[np.ndarray] = []
    stats_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []

    for eps_i, epsilon in enumerate(epsilons):
        eps = float(epsilon)
        rng = np.random.default_rng(int(seed) + 9001 + 131 * eps_i + (0 if split_name == "train" else 777_777))
        attack_res = run_query_sparse_hillclimb(
            x_orig=x_benign.astype(np.float32, copy=False),
            baseline_margin=baseline_margin,
            booster=booster,
            threshold=float(threshold),
            profile=profile,
            epsilon=eps,
            objective="maximize",
            query_budget=int(query_budget),
            max_steps=int(query_max_steps),
            candidates_per_step=int(query_candidates_per_step),
            feature_subset_size=int(query_feature_subset_size),
            rng=rng,
            progress_prefix=f"hardneg[{split_name}] eps={eps}",
            start_ts=start_ts,
        )

        x_adv = attack_res["x_adv"].astype(np.float32, copy=False)
        queries_used = attack_res["queries_used"].astype(np.int32, copy=False)
        final_margin = attack_res["final_margin"].astype(np.float64, copy=False)
        success = attack_res["success"].astype(bool, copy=False)
        changed_mask = (np.max(np.abs(x_adv.astype(np.float64) - x_benign.astype(np.float64)), axis=1) > 1e-9)
        if np.any(changed_mask):
            violations = summarize_realistic_violations(
                x_adv=x_adv[changed_mask],
                x_orig=x_benign[changed_mask],
                profile=profile,
                epsilon=eps,
            )
        else:
            violations = _empty_violation_dict()

        if (
            int(violations["below_lower_violations"]) > 0
            or int(violations["above_upper_violations"]) > 0
            or int(violations["locked_change_rows"]) > 0
            or int(violations["relation_violation_rows"]) > 0
        ):
            raise RuntimeError(
                f"Constraint integrity failure in hardneg campaign split={split_name} eps={eps}: {violations}"
            )
        if int(np.max(queries_used)) > int(query_budget):
            raise RuntimeError(
                f"Query budget exceeded in split={split_name} eps={eps}: "
                f"max_queries={int(np.max(queries_used))} budget={int(query_budget)}"
            )

        flip_mask = baseline_benign & (final_margin >= 0.0) & success
        selected_mask = flip_mask.copy()
        topk_added = 0
        if int(extra_topk_per_epsilon) > 0:
            candidate_idx = np.flatnonzero(baseline_benign & (~flip_mask))
            if candidate_idx.size > 0:
                k = min(int(extra_topk_per_epsilon), int(candidate_idx.size))
                if k > 0:
                    if k >= candidate_idx.size:
                        chosen_idx = candidate_idx
                    else:
                        scores = final_margin[candidate_idx]
                        top_local = np.argpartition(scores, -k)[-k:]
                        chosen_idx = candidate_idx[top_local]
                    selected_mask[chosen_idx] = True
                    topk_added = int(chosen_idx.size)
        flipped_chunks.append(x_adv[flip_mask].copy())
        selected_chunks.append(x_adv[selected_mask].copy())
        all_adv_chunks.append(x_adv.copy())
        all_eps_chunks.append(np.full(n_rows, eps, dtype=np.float64))

        attacked_pred = (final_margin >= 0.0).astype(np.int8)
        attacked_fpr = float(np.mean(attacked_pred.astype(np.float64)))
        stats_rows.append(
            {
                "split": split_name,
                "epsilon": eps,
                "targeted_rows": int(n_rows),
                "baseline_false_alarm_rows": int(np.sum(~baseline_benign)),
                "attack_successes_any": int(np.sum(success)),
                "flip_rows": int(np.sum(flip_mask)),
                "flip_rate": float(np.mean(flip_mask.astype(np.float64))),
                "selected_rows": int(np.sum(selected_mask)),
                "selected_rate": float(np.mean(selected_mask.astype(np.float64))),
                "selected_topk_added_rows": int(topk_added),
                "attacked_pred_attack_rate": attacked_fpr,
                "queries_total": int(np.sum(queries_used)),
                "queries_mean": float(np.mean(queries_used.astype(np.float64))),
                "queries_p95": float(np.percentile(queries_used.astype(np.float64), 95.0)),
                "accepted_moves_mean": float(attack_res["summary"]["accepted_moves_mean"]),
                **{k: (float(v) if k == "locked_change_max_abs" else int(v)) for k, v in violations.items()},
            }
        )

        flip_idx = np.flatnonzero(flip_mask)
        if flip_idx.size > 0:
            x0 = x_benign[flip_idx].astype(np.float64, copy=False)
            xa = x_adv[flip_idx].astype(np.float64, copy=False)
            delta = xa - x0
            l0 = np.sum(np.abs(delta) > 1e-9, axis=1)
            l2 = np.sqrt(np.sum(np.square(delta), axis=1))
            linf = np.max(np.abs(delta), axis=1)
            for rel_i, row_i in enumerate(flip_idx.tolist()):
                preview_rows.append(
                    {
                        "split": split_name,
                        "epsilon": eps,
                        "source_relpath": str(src_relpath[row_i]),
                        "source_row_index": int(src_row_index[row_i]),
                        "baseline_margin": float(baseline_margin[row_i]),
                        "adv_margin": float(final_margin[row_i]),
                        "queries_used": int(queries_used[row_i]),
                        "l0": int(l0[rel_i]),
                        "l2": float(l2[rel_i]),
                        "linf": float(linf[rel_i]),
                    }
                )

    flipped_adv = np.vstack(flipped_chunks) if flipped_chunks else np.empty((0, n_features), dtype=np.float32)
    selected_adv = np.vstack(selected_chunks) if selected_chunks else np.empty((0, n_features), dtype=np.float32)
    all_adv = np.vstack(all_adv_chunks) if all_adv_chunks else np.empty((0, n_features), dtype=np.float32)
    all_adv_eps = np.concatenate(all_eps_chunks) if all_eps_chunks else np.empty(0, dtype=np.float64)
    return {
        "flipped_adv": flipped_adv.astype(np.float32, copy=False),
        "selected_adv": selected_adv.astype(np.float32, copy=False),
        "all_adv": all_adv.astype(np.float32, copy=False),
        "all_adv_eps": all_adv_eps.astype(np.float64, copy=False),
        "stats_rows": stats_rows,
        "preview_rows": preview_rows,
    }


def _run_malicious_query_eval(
    *,
    split_name: str,
    x_malicious: np.ndarray,
    src_relpath: np.ndarray,
    src_row_index: np.ndarray,
    booster: xgb.Booster,
    threshold: float,
    profile: Dict[str, Any],
    epsilon: float,
    query_budget: int,
    query_max_steps: int,
    query_candidates_per_step: int,
    query_feature_subset_size: int,
    seed: int,
    start_ts: float,
) -> Dict[str, Any]:
    n_rows, n_features = x_malicious.shape
    if n_rows <= 0:
        return {
            "x_adv": np.empty((0, n_features), dtype=np.float32),
            "stats_row": {
                "split": split_name,
                "epsilon": float(epsilon),
                "targeted_rows": 0,
                "baseline_detected_rows": 0,
                "evasion_flip_rows": 0,
                "evasion_flip_rate": float("nan"),
                "robust_recall_under_attack": float("nan"),
                "queries_total": 0,
                "queries_mean": float("nan"),
                "queries_p95": float("nan"),
                "accepted_moves_mean": float("nan"),
            },
            "preview_rows": [],
        }

    baseline_margin = score_margin_batch(booster, x_malicious.astype(np.float32, copy=False), threshold=float(threshold))
    baseline_detected = baseline_margin >= 0.0
    rng = np.random.default_rng(int(seed))
    attack_res = run_query_sparse_hillclimb(
        x_orig=x_malicious.astype(np.float32, copy=False),
        baseline_margin=baseline_margin,
        booster=booster,
        threshold=float(threshold),
        profile=profile,
        epsilon=float(epsilon),
        objective="minimize",
        query_budget=int(query_budget),
        max_steps=int(query_max_steps),
        candidates_per_step=int(query_candidates_per_step),
        feature_subset_size=int(query_feature_subset_size),
        rng=rng,
        progress_prefix=f"hardneg_mal[{split_name}] eps={float(epsilon)}",
        start_ts=start_ts,
    )

    x_adv = attack_res["x_adv"].astype(np.float32, copy=False)
    queries_used = attack_res["queries_used"].astype(np.int32, copy=False)
    final_margin = attack_res["final_margin"].astype(np.float64, copy=False)
    success = attack_res["success"].astype(bool, copy=False)
    changed_mask = (np.max(np.abs(x_adv.astype(np.float64) - x_malicious.astype(np.float64)), axis=1) > 1e-9)
    if np.any(changed_mask):
        violations = summarize_realistic_violations(
            x_adv=x_adv[changed_mask],
            x_orig=x_malicious[changed_mask],
            profile=profile,
            epsilon=float(epsilon),
        )
    else:
        violations = _empty_violation_dict()

    if (
        int(violations["below_lower_violations"]) > 0
        or int(violations["above_upper_violations"]) > 0
        or int(violations["locked_change_rows"]) > 0
        or int(violations["relation_violation_rows"]) > 0
    ):
        raise RuntimeError(
            f"Constraint integrity failure in malicious campaign split={split_name} eps={epsilon}: {violations}"
        )
    if int(np.max(queries_used)) > int(query_budget):
        raise RuntimeError(
            f"Query budget exceeded in malicious campaign split={split_name} eps={epsilon}: "
            f"max_queries={int(np.max(queries_used))} budget={int(query_budget)}"
        )

    adv_attack_pred = final_margin >= 0.0
    flip_mask = baseline_detected & (~adv_attack_pred) & success

    stats_row: Dict[str, Any] = {
        "split": split_name,
        "epsilon": float(epsilon),
        "targeted_rows": int(n_rows),
        "baseline_detected_rows": int(np.sum(baseline_detected)),
        "evasion_flip_rows": int(np.sum(flip_mask)),
        "evasion_flip_rate": float(np.mean(flip_mask.astype(np.float64))),
        "robust_recall_under_attack": float(np.mean(adv_attack_pred.astype(np.float64))),
        "queries_total": int(np.sum(queries_used)),
        "queries_mean": float(np.mean(queries_used.astype(np.float64))),
        "queries_p95": float(np.percentile(queries_used.astype(np.float64), 95.0)),
        "accepted_moves_mean": float(attack_res["summary"]["accepted_moves_mean"]),
        **{k: (float(v) if k == "locked_change_max_abs" else int(v)) for k, v in violations.items()},
    }

    preview_rows: List[Dict[str, Any]] = []
    preview_idx = np.flatnonzero(flip_mask)
    if preview_idx.size > 0:
        x0 = x_malicious[preview_idx].astype(np.float64, copy=False)
        xa = x_adv[preview_idx].astype(np.float64, copy=False)
        delta = xa - x0
        l0 = np.sum(np.abs(delta) > 1e-9, axis=1)
        l2 = np.sqrt(np.sum(np.square(delta), axis=1))
        linf = np.max(np.abs(delta), axis=1)
        for rel_i, row_i in enumerate(preview_idx.tolist()):
            preview_rows.append(
                {
                    "split": split_name,
                    "epsilon": float(epsilon),
                    "source_relpath": str(src_relpath[row_i]),
                    "source_row_index": int(src_row_index[row_i]),
                    "baseline_margin": float(baseline_margin[row_i]),
                    "adv_margin": float(final_margin[row_i]),
                    "queries_used": int(queries_used[row_i]),
                    "l0": int(l0[rel_i]),
                    "l2": float(l2[rel_i]),
                    "linf": float(linf[rel_i]),
                }
            )

    return {
        "x_adv": x_adv.astype(np.float32, copy=False),
        "stats_row": stats_row,
        "preview_rows": preview_rows,
    }


def _load_wifi_xgb_hparams(base_run_dir: Path) -> Dict[str, Any]:
    path = base_run_dir / "best_hparams.json"
    if not path.exists():
        raise RuntimeError(f"Missing best_hparams.json in base run: {base_run_dir}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    h = payload.get("wifi", {}).get("xgboost", {})
    if not isinstance(h, dict) or not h:
        raise RuntimeError("Could not resolve WiFi XGBoost hyperparameters from best_hparams.json")
    return h


def _fit_wifi_xgb(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hparams: Dict[str, Any],
    seed: int,
    requested_device: str,
    start_ts: float,
) -> Tuple[xgb.Booster, str]:
    device_attempts = [requested_device]
    if requested_device == "cuda":
        device_attempts.append("cpu")

    last_error: Optional[Exception] = None
    for device in device_attempts:
        num_boost_round = int(hparams.get("n_estimators", 1500))
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": device,
            "eval_metric": "aucpr",
            "seed": int(seed),
            "max_depth": int(hparams.get("max_depth", 6)),
            "eta": float(hparams.get("learning_rate", 0.05)),
            "subsample": float(hparams.get("subsample", 0.8)),
            "colsample_bytree": float(hparams.get("colsample_bytree", 0.8)),
            "min_child_weight": float(hparams.get("min_child_weight", 1.0)),
            "lambda": float(hparams.get("reg_lambda", 1.0)),
            "alpha": float(hparams.get("reg_alpha", 0.0)),
            "gamma": float(hparams.get("gamma", 0.0)),
            "max_bin": int(hparams.get("max_bin", 256)),
        }
        try:
            log_progress(f"training WiFi XGBoost with device={device}", start_ts=start_ts)
            dtrain = xgb.DMatrix(
                x_train.astype(np.float32, copy=False),
                label=y_train.astype(np.float32, copy=False),
                weight=sample_weight.astype(np.float32, copy=False),
            )
            dval = xgb.DMatrix(
                x_val.astype(np.float32, copy=False),
                label=y_val.astype(np.float32, copy=False),
            )
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "val")],
                verbose_eval=100,
            )
            return booster, device
        except xgb.core.XGBoostError as ex:
            last_error = ex
            if device == "cuda":
                log_progress(f"cuda training failed, retrying on cpu: {ex}", start_ts=start_ts)
                continue
            raise
    if last_error is not None:
        raise RuntimeError(f"Failed to train WiFi XGBoost: {last_error}") from last_error
    raise RuntimeError("Unexpected training failure")

def main() -> None:
    start_ts = time.time()
    args = parse_args()
    epsilons = parse_float_csv(args.hardneg_epsilons)

    np.random.seed(int(args.seed))
    base_run_dir = Path(args.base_run_dir).resolve() if args.base_run_dir else discover_latest_hpo_run(Path("reports"))
    if base_run_dir is None:
        raise RuntimeError("Could not resolve base run directory.")
    base_run_dir = Path(base_run_dir).resolve()
    train_csv = Path(args.train_csv).resolve()
    _ = Path(args.test_csv).resolve()                             
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    log_progress(f"base_run_dir={base_run_dir}", start_ts=start_ts)
    log_progress(f"train_csv={train_csv}", start_ts=start_ts)
    log_progress(f"hardneg_epsilons={epsilons}", start_ts=start_ts)

    feature_columns = load_feature_columns(base_run_dir)
    models_by_protocol = load_xgb_models_by_protocol(base_run_dir)
    resolved_device = configure_boosters_device(
        models_by_protocol=models_by_protocol,
        feature_count=len(feature_columns),
        requested_device=args.xgb_device,
        start_ts=start_ts,
    )
    thresholds_by_protocol = load_thresholds_by_protocol(base_run_dir)
    wifi_threshold_base = float(thresholds_by_protocol.get("wifi", 0.5))
    if "wifi" not in models_by_protocol:
        raise RuntimeError("WiFi XGBoost model not found in base run.")
    wifi_base_booster = models_by_protocol["wifi"]
    wifi_hparams = _load_wifi_xgb_hparams(base_run_dir)

    log_progress(f"resolved_booster_device={resolved_device}", start_ts=start_ts)
    log_progress(f"wifi_base_threshold={wifi_threshold_base:.6f}", start_ts=start_ts)

    wifi_df = _load_wifi_train_rows(
        train_csv=train_csv,
        feature_columns=feature_columns,
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )
    log_progress(f"wifi rows loaded: {len(wifi_df)}", start_ts=start_ts)

    val_mask, fallback_events = _build_protocol_class_val_mask(
        wifi_df[["source_relpath", "source_row_index", "protocol_hint", "label"]],
        seed=int(args.seed),
        val_mod=int(args.val_mod),
        allow_row_fallback=True,
    )
    log_progress(
        f"wifi split ready: train={int((~val_mask).sum())}, val={int(val_mask.sum())}, fallbacks={len(fallback_events)}",
        start_ts=start_ts,
    )

    x_all = wifi_df.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)
    y_all = normalize_label_series(wifi_df["label"])
    src_relpath_all = wifi_df["source_relpath"].astype(str).to_numpy(dtype=object, copy=False)
    src_row_index_all = pd.to_numeric(wifi_df["source_row_index"], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)

    x_train = x_all[~val_mask]
    y_train = y_all[~val_mask]
    src_train_relpath = src_relpath_all[~val_mask]
    src_train_row_index = src_row_index_all[~val_mask]

    x_val = x_all[val_mask]
    y_val = y_all[val_mask]
    src_val_relpath = src_relpath_all[val_mask]
    src_val_row_index = src_row_index_all[val_mask]

    train_benign_idx_all = np.flatnonzero(y_train == 0)
    val_benign_idx_all = np.flatnonzero(y_val == 0)
    val_malicious_idx_all = np.flatnonzero(y_val == 1)
    if train_benign_idx_all.size <= 0 or val_benign_idx_all.size <= 0:
        raise RuntimeError(
            f"Need benign rows in both train and val splits. train_benign={train_benign_idx_all.size}, "
            f"val_benign={val_benign_idx_all.size}"
        )
    if val_malicious_idx_all.size <= 0:
        raise RuntimeError("Need malicious rows in validation split for robust threshold calibration.")

    rng = np.random.default_rng(int(args.seed))
    train_benign_pick = _sample_indices(train_benign_idx_all.size, int(args.hardneg_train_benign_sample), rng)
    val_benign_pick = _sample_indices(val_benign_idx_all.size, int(args.hardneg_val_benign_sample), rng)
    val_malicious_pick = _sample_indices(val_malicious_idx_all.size, int(args.val_malicious_sample), rng)
    train_benign_idx = train_benign_idx_all[train_benign_pick]
    val_benign_idx = val_benign_idx_all[val_benign_pick]
    val_malicious_idx = val_malicious_idx_all[val_malicious_pick]
    log_progress(
        f"benign samples chosen: train={train_benign_idx.size}/{train_benign_idx_all.size}, "
        f"val={val_benign_idx.size}/{val_benign_idx_all.size}",
        start_ts=start_ts,
    )
    log_progress(
        f"malicious val sample chosen: val={val_malicious_idx.size}/{val_malicious_idx_all.size}",
        start_ts=start_ts,
    )

                                                         
    if int(args.constraint_sample_size) > 0 and x_train.shape[0] > int(args.constraint_sample_size):
        sample_idx = rng.choice(x_train.shape[0], size=int(args.constraint_sample_size), replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(x_train.shape[0], dtype=np.int64)
    constraint_sample_df = pd.DataFrame(x_train[sample_idx], columns=feature_columns)
    constraint_sample_df.insert(0, "label", y_train[sample_idx].astype(np.int8))
    constraint_sample_df.insert(0, "protocol_hint", "wifi")

    constraints, constraints_summary_df = build_constraints(
        train_sample_df=constraint_sample_df,
        feature_columns=feature_columns,
        protocols=["wifi"],
        percentile_lower=float(args.percentile_lower),
        percentile_upper=float(args.percentile_upper),
    )
    realism_profiles, realism_profile_json = build_realism_profiles(
        train_sample_df=constraint_sample_df,
        feature_columns=feature_columns,
        protocols=["wifi"],
        constraints=constraints,
        relation_q_low=float(args.relation_quantile_low),
        relation_q_high=float(args.relation_quantile_high),
        relative_cap_default=float(args.relative_cap_default),
        relative_cap_rate_group=args.relative_cap_rate_group,
        relative_cap_size_group=args.relative_cap_size_group,
        relative_cap_time_group=args.relative_cap_time_group,
    )
    wifi_profile = realism_profiles["wifi"]
    log_progress("realism profile ready for wifi", start_ts=start_ts)

    query_budget_benign = int(args.hardneg_query_budget)
    query_max_steps_benign = int(args.hardneg_query_max_steps)
    query_candidates_benign = int(args.hardneg_candidates_per_step)
    query_subset_benign = int(args.hardneg_feature_subset_size)

    query_budget_mal = (
        int(args.val_malicious_query_budget)
        if args.val_malicious_query_budget is not None
        else query_budget_benign
    )
    query_max_steps_mal = (
        int(args.val_malicious_query_max_steps)
        if args.val_malicious_query_max_steps is not None
        else query_max_steps_benign
    )
    query_candidates_mal = (
        int(args.val_malicious_candidates_per_step)
        if args.val_malicious_candidates_per_step is not None
        else query_candidates_benign
    )
    query_subset_mal = (
        int(args.val_malicious_feature_subset_size)
        if args.val_malicious_feature_subset_size is not None
        else query_subset_benign
    )

    est_queries_train = int(train_benign_idx.size) * len(epsilons) * query_budget_benign
    est_queries_val = int(val_benign_idx.size) * len(epsilons) * query_budget_benign
    log_progress(
        f"query workload estimate (benign campaigns): train<= {est_queries_train:,} queries, val<= {est_queries_val:,} queries",
        start_ts=start_ts,
    )

    train_campaign = _run_benign_query_campaign(
        split_name="train",
        x_benign=x_train[train_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_train_relpath[train_benign_idx],
        src_row_index=src_train_row_index[train_benign_idx],
        booster=wifi_base_booster,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        extra_topk_per_epsilon=int(args.hardneg_extra_topk_per_epsilon),
        seed=int(args.seed),
        start_ts=start_ts,
    )
    val_campaign = _run_benign_query_campaign(
        split_name="val",
        x_benign=x_val[val_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_benign_idx],
        src_row_index=src_val_row_index[val_benign_idx],
        booster=wifi_base_booster,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        extra_topk_per_epsilon=int(args.hardneg_extra_topk_per_epsilon),
        seed=int(args.seed) + 91_001,
        start_ts=start_ts,
    )

    n_hardneg_flips = int(train_campaign["flipped_adv"].shape[0])
    x_hardneg = train_campaign["selected_adv"]
    n_hardneg = int(x_hardneg.shape[0])
    log_progress(
        (
            "hard negatives collected: "
            f"train_flips={n_hardneg_flips}, train_selected={n_hardneg}, "
            f"extra_topk_per_epsilon={int(args.hardneg_extra_topk_per_epsilon)}"
        ),
        start_ts=start_ts,
    )
    if n_hardneg <= 0:
        raise RuntimeError(
            "No train hard negatives were generated from benign query campaign. "
            "Increase benign sample size/query budget or relax constraints and retry."
        )

    x_train_aug = np.vstack([x_train, x_hardneg.astype(np.float32, copy=False)])
    y_train_aug = np.concatenate([y_train, np.zeros(n_hardneg, dtype=np.int8)], axis=0)
    sample_weight = np.ones(y_train_aug.shape[0], dtype=np.float32)
    sample_weight[-n_hardneg:] = float(args.hardneg_weight)

    wifi_model, trained_device = _fit_wifi_xgb(
        x_train=x_train_aug,
        y_train=y_train_aug,
        sample_weight=sample_weight,
        x_val=x_val,
        y_val=y_val,
        hparams=wifi_hparams,
        seed=int(args.seed),
        requested_device=str(args.xgb_device),
        start_ts=start_ts,
    )
    dval_clean = xgb.DMatrix(x_val.astype(np.float32, copy=False))
    p_val_clean = wifi_model.predict(dval_clean).astype(np.float64, copy=False)
    p_val_adv_benign = (
        wifi_model.predict(xgb.DMatrix(val_campaign["all_adv"].astype(np.float32, copy=False))).astype(np.float64, copy=False)
        if val_campaign["all_adv"].shape[0] > 0
        else np.empty(0, dtype=np.float64)
    )

    mal_eval_epsilon = float(np.max(np.array(epsilons, dtype=np.float64)))
    mal_eval = _run_malicious_query_eval(
        split_name="val",
        x_malicious=x_val[val_malicious_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_malicious_idx],
        src_row_index=src_val_row_index[val_malicious_idx],
        booster=wifi_model,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilon=mal_eval_epsilon,
        query_budget=query_budget_mal,
        query_max_steps=query_max_steps_mal,
        query_candidates_per_step=query_candidates_mal,
        query_feature_subset_size=query_subset_mal,
        seed=int(args.seed) + 303_031,
        start_ts=start_ts,
    )
    p_val_adv_malicious = (
        wifi_model.predict(xgb.DMatrix(mal_eval["x_adv"].astype(np.float32, copy=False))).astype(np.float64, copy=False)
        if mal_eval["x_adv"].shape[0] > 0
        else np.empty(0, dtype=np.float64)
    )

    threshold_sel = _select_threshold(
        y_clean=y_val,
        p_clean=p_val_clean,
        p_adv_benign=p_val_adv_benign,
        p_adv_malicious=p_val_adv_malicious,
        recall_target=float(args.wifi_recall_target),
        adv_malicious_recall_target=float(args.threshold_adv_malicious_recall_target),
        adv_shortfall_weight=float(args.threshold_fallback_adv_shortfall_weight),
        grid_size=int(args.threshold_grid_size),
        base_threshold=wifi_threshold_base,
    )
    wifi_threshold_new = float(threshold_sel["selected_threshold"])

    val_clean_pred_new = (p_val_clean >= wifi_threshold_new).astype(np.int8)
    val_clean_pred_old = (p_val_clean >= wifi_threshold_base).astype(np.int8)
    val_clean_metrics_new = _binary_metrics(y_val, val_clean_pred_new)
    val_clean_metrics_old = _binary_metrics(y_val, val_clean_pred_old)
    val_adv_fpr_new = float(np.mean((p_val_adv_benign >= wifi_threshold_new).astype(np.float64))) if p_val_adv_benign.size > 0 else float("nan")
    val_adv_fpr_old = float(np.mean((p_val_adv_benign >= wifi_threshold_base).astype(np.float64))) if p_val_adv_benign.size > 0 else float("nan")
    val_adv_mal_recall_new = (
        float(np.mean((p_val_adv_malicious >= wifi_threshold_new).astype(np.float64)))
        if p_val_adv_malicious.size > 0
        else float("nan")
    )
    val_adv_mal_recall_old = (
        float(np.mean((p_val_adv_malicious >= wifi_threshold_base).astype(np.float64)))
        if p_val_adv_malicious.size > 0
        else float("nan")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run_dir = out_root / f"{base_run_dir.name}_wifi_robust_v1_{ts}"
    out_run_dir.mkdir(parents=True, exist_ok=True)
    (out_run_dir / "models").mkdir(parents=True, exist_ok=True)

    for name in [
        "metrics_summary.json",
        "metrics_summary.csv",
        "metrics_summary_per_protocol_models.csv",
        "best_hparams.json",
        "ensemble_weights.json",
        "hpo_trials.csv",
        "RUN_SUMMARY.txt",
        "slice_metrics_protocol.csv",
        "slice_metrics_attack_family.csv",
    ]:
        src = base_run_dir / name
        dst = out_run_dir / name
        if src.exists():
            shutil.copy2(src, dst)

    base_models_dir = base_run_dir / "models"
    out_models_dir = out_run_dir / "models"
    for model_file in base_models_dir.glob("*"):
        if model_file.is_file():
            shutil.copy2(model_file, out_models_dir / model_file.name)
    wifi_model_path = out_models_dir / "wifi__xgboost_tuned.json"
    wifi_model.save_model(str(wifi_model_path))

    thresholds_new = dict(thresholds_by_protocol)
    thresholds_new["wifi"] = wifi_threshold_new
    thresholds_path = out_run_dir / "thresholds_by_protocol.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in sorted(thresholds_new.items())}, f, indent=2)

    hardneg_stats_rows = train_campaign["stats_rows"] + val_campaign["stats_rows"] + [mal_eval["stats_row"]]
    hardneg_stats_df = pd.DataFrame(hardneg_stats_rows)
    hardneg_stats_path = out_run_dir / "hard_negative_stats.csv"
    hardneg_stats_df.to_csv(hardneg_stats_path, index=False)

    preview_rows = train_campaign["preview_rows"] + val_campaign["preview_rows"] + mal_eval["preview_rows"]
    if preview_rows:
        preview_df = pd.DataFrame(preview_rows).head(max(1, int(args.preview_max_rows)))
    else:
        preview_df = pd.DataFrame(
            columns=[
                "split",
                "epsilon",
                "source_relpath",
                "source_row_index",
                "baseline_margin",
                "adv_margin",
                "queries_used",
                "l0",
                "l2",
                "linf",
            ]
        )
    preview_path = out_run_dir / "hard_negative_preview.csv"
    preview_df.to_csv(preview_path, index=False)

    constraints_summary_path = out_run_dir / "hardening_constraints_summary.csv"
    constraints_summary_df.to_csv(constraints_summary_path, index=False)
    realism_profile_path = out_run_dir / "hardening_realism_profile.json"
    with realism_profile_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(realism_profile_json), f, indent=2)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "base_run_dir": str(base_run_dir),
        "output_run_dir": str(out_run_dir),
        "trained_device": str(trained_device),
        "xgb_booster_inference_device": str(resolved_device),
        "feature_count": int(len(feature_columns)),
        "wifi_hparams": to_jsonable(wifi_hparams),
        "split": {
            "train_rows": int(x_train.shape[0]),
            "val_rows": int(x_val.shape[0]),
            "train_benign_rows": int(train_benign_idx_all.size),
            "val_benign_rows": int(val_benign_idx_all.size),
            "val_malicious_rows": int(val_malicious_idx_all.size),
            "train_benign_sampled": int(train_benign_idx.size),
            "val_benign_sampled": int(val_benign_idx.size),
            "val_malicious_sampled": int(val_malicious_idx.size),
            "fallback_events": to_jsonable(fallback_events),
        },
        "hard_negative": {
            "train_flip_rows": int(n_hardneg_flips),
            "train_selected_rows": int(n_hardneg),
            "train_flip_rows_strict": int(n_hardneg_flips),
            "hardneg_weight": float(args.hardneg_weight),
            "extra_topk_per_epsilon": int(args.hardneg_extra_topk_per_epsilon),
            "train_campaign_rows": to_jsonable(train_campaign["stats_rows"]),
            "val_campaign_rows": to_jsonable(val_campaign["stats_rows"]),
            "val_malicious_campaign": to_jsonable(mal_eval["stats_row"]),
        },
        "thresholds": {
            "wifi_base": float(wifi_threshold_base),
            "wifi_new": float(wifi_threshold_new),
            "selection": to_jsonable(threshold_sel),
            "all_protocol_thresholds_new": {str(k): float(v) for k, v in sorted(thresholds_new.items())},
        },
        "validation_metrics": {
            "clean_old_threshold": val_clean_metrics_old,
            "clean_new_threshold": val_clean_metrics_new,
            "attacked_benign_fpr_old_threshold": float(val_adv_fpr_old),
            "attacked_benign_fpr_new_threshold": float(val_adv_fpr_new),
            "attacked_malicious_recall_old_threshold": float(val_adv_mal_recall_old),
            "attacked_malicious_recall_new_threshold": float(val_adv_mal_recall_new),
            "delta_clean_f1": float(val_clean_metrics_new["f1"] - val_clean_metrics_old["f1"]),
            "delta_clean_recall": float(val_clean_metrics_new["recall"] - val_clean_metrics_old["recall"]),
        },
        "files": {
            "wifi_model": str(wifi_model_path),
            "thresholds_by_protocol": str(thresholds_path),
            "hard_negative_stats_csv": str(hardneg_stats_path),
            "hard_negative_preview_csv": str(preview_path),
            "hardening_constraints_summary_csv": str(constraints_summary_path),
            "hardening_realism_profile_json": str(realism_profile_path),
        },
    }
    summary_path = out_run_dir / "hardening_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    log_progress(f"saved hardening run: {out_run_dir}", start_ts=start_ts)
    log_progress(f"saved summary: {summary_path}", start_ts=start_ts)
    print(str(out_run_dir))


if __name__ == "__main__":
    main()
