from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


PROTOCOLS = ("wifi", "mqtt", "bluetooth")


def interleave_row_groups(groups: List[pd.DataFrame], shuffle_seed: int) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    rng = np.random.default_rng(int(shuffle_seed))
    order = np.arange(len(groups), dtype=int)
    rng.shuffle(order)
    groups = [groups[i] for i in order]

    max_len = max(len(g) for g in groups)
    out_parts = []
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                out_parts.append(g.iloc[[i]])
    return pd.concat(out_parts, ignore_index=True)


def interleave_by_source(df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    if "source_relpath" not in df.columns:
        return df.sample(frac=1.0, random_state=int(shuffle_seed)).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby("source_relpath", sort=False)]
    if not groups:
        return df.reset_index(drop=True)
    return interleave_row_groups(groups, shuffle_seed=int(shuffle_seed))


def interleave_by_protocol_source(df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    if "protocol_hint" not in df.columns:
        return interleave_by_source(df, shuffle_seed=int(shuffle_seed))
    protocol_streams = []
    for _, proto_df in df.groupby("protocol_hint", sort=False):
        protocol_streams.append(interleave_by_source(proto_df.reset_index(drop=True), shuffle_seed=int(shuffle_seed)))
    if not protocol_streams:
        return df.reset_index(drop=True)
    return interleave_row_groups(protocol_streams, shuffle_seed=int(shuffle_seed))


def reorder_rows(df: pd.DataFrame, replay_order: str, shuffle_seed: int) -> pd.DataFrame:
    if df.empty or replay_order == "sequential":
        return df.reset_index(drop=True)
    if replay_order == "shuffle":
        return df.sample(frac=1.0, random_state=int(shuffle_seed)).reset_index(drop=True)
    if replay_order == "interleave-source":
        return interleave_by_source(df, shuffle_seed=int(shuffle_seed))
    if replay_order == "interleave-protocol-source":
        return interleave_by_protocol_source(df, shuffle_seed=int(shuffle_seed))
    return df.reset_index(drop=True)


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def protocol_slug(protocol: object) -> str:
    raw = safe_str(protocol).strip().lower()
    raw = "".join(ch if ch.isalnum() else "_" for ch in raw)
    raw = raw.strip("_")
    return raw if raw else "unknown"


def discover_latest_matrix_run(reports_dir: str | Path = "reports") -> Path | None:
    base = Path(reports_dir)
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.glob("*_protocol_multimodel_robust_matrix_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_feature_columns(base_run_dir: Path) -> List[str]:
    metrics_path = base_run_dir / "metrics_summary.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cols = payload.get("feature_columns", [])
    if not isinstance(cols, list) or not cols:
        raise RuntimeError(f"Could not read feature columns from {metrics_path}")
    return [safe_str(col) for col in cols]


def load_catboost_e_models_by_protocol(matrix_run_dir: Path) -> Dict[str, CatBoostClassifier]:
    models: Dict[str, CatBoostClassifier] = {}
    for proto in PROTOCOLS:
        model_path = (
            matrix_run_dir
            / "candidates"
            / f"coarse_{proto}_catboost_E_seed_42"
            / "models"
            / f"{proto}__catboost__E.cbm"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing CatBoost E model for {proto}: {model_path}")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        models[proto] = model
    return models


def load_thresholds_by_policy(matrix_run_dir: Path, policy: str) -> Dict[str, float]:
    summary_path = matrix_run_dir / "catboost_E_threshold_tiny_tune_surrogate_guard_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        row = summary_df[summary_df["policy"].astype(str) == policy]
        if not row.empty:
            r0 = row.iloc[0]
            out = {
                "wifi": safe_float(r0.get("wifi_threshold"), math.nan),
                "mqtt": safe_float(r0.get("mqtt_threshold"), math.nan),
                "bluetooth": safe_float(r0.get("bluetooth_threshold"), math.nan),
            }
            if all(np.isfinite(v) for v in out.values()):
                return out

    out: Dict[str, float] = {}
    for proto in PROTOCOLS:
        candidate_summary = (
            matrix_run_dir
            / "candidates"
            / f"coarse_{proto}_catboost_E_seed_42"
            / "candidate_summary.json"
        )
        if not candidate_summary.exists():
            raise FileNotFoundError(f"Missing candidate summary for {proto}: {candidate_summary}")
        with candidate_summary.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        out[proto] = safe_float(payload.get("selected_threshold"), 0.5)
    return out


def resolve_prediction_task_type(
    models_by_protocol: Dict[str, CatBoostClassifier],
    feature_count: int,
    inference_device: str,
) -> Tuple[str, str]:
    req = safe_str(inference_device).strip().lower() or "auto"
    if req == "cpu":
        return "CPU", ""

    probe = Pool(np.zeros((1, int(feature_count)), dtype=np.float32))
    sample_model = next(iter(models_by_protocol.values()))
    try:
        sample_model.predict_proba(probe, task_type="GPU")
        return "GPU", ""
    except Exception as exc:
        msg = safe_str(exc)
        if req == "gpu":
            return "CPU", f"gpu_requested_fallback_cpu: {msg}"
        return "CPU", f"auto_cpu_fallback: {msg}"


def normalize_protocol_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .map(protocol_slug)
    )


def normalize_label_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8).clip(0, 1)


def read_csv_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def _jsonable_cache_config(
    train_csv: Path,
    test_csv: Path,
    feature_columns: List[str],
    total_rows: int,
    attack_ratio: float,
    seed: int,
    replay_order: str,
    shuffle_seed: int,
) -> Dict[str, object]:
    train_stat = train_csv.stat()
    test_stat = test_csv.stat()
    return {
        "cache_version": "catboost_e_demo_cache_v2_full_sweep_dynamic",
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_size": int(train_stat.st_size),
        "test_size": int(test_stat.st_size),
        "train_mtime_ns": int(train_stat.st_mtime_ns),
        "test_mtime_ns": int(test_stat.st_mtime_ns),
        "feature_count": int(len(feature_columns)),
        "total_rows": int(total_rows),
        "attack_ratio": float(round(float(attack_ratio), 8)),
        "seed": int(seed),
        "replay_order": str(replay_order),
        "shuffle_seed": int(shuffle_seed),
    }


def compute_demo_cache_key(
    train_csv: Path,
    test_csv: Path,
    feature_columns: List[str],
    total_rows: int,
    attack_ratio: float,
    seed: int,
    replay_order: str,
    shuffle_seed: int,
) -> str:
    payload = _jsonable_cache_config(
        train_csv=train_csv,
        test_csv=test_csv,
        feature_columns=feature_columns,
        total_rows=total_rows,
        attack_ratio=attack_ratio,
        seed=seed,
        replay_order=replay_order,
        shuffle_seed=shuffle_seed,
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_demo_cache_path(
    cache_dir: Path,
    train_csv: Path,
    test_csv: Path,
    feature_columns: List[str],
    total_rows: int,
    attack_ratio: float,
    seed: int,
    replay_order: str,
    shuffle_seed: int,
) -> Path:
    key = compute_demo_cache_key(
        train_csv=train_csv,
        test_csv=test_csv,
        feature_columns=feature_columns,
        total_rows=total_rows,
        attack_ratio=attack_ratio,
        seed=seed,
        replay_order=replay_order,
        shuffle_seed=shuffle_seed,
    )
    return cache_dir / f"catboost_e_demo_subset_{key}.pkl"


def load_demo_cache(cache_path: Path) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = pd.read_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    df = payload.get("df")
    meta = payload.get("subset_meta")
    global_exp = payload.get("global_explanations")
    if not isinstance(df, pd.DataFrame) or not isinstance(meta, dict) or not isinstance(global_exp, dict):
        return None
    return df, meta, global_exp


def save_demo_cache(
    cache_path: Path,
    df: pd.DataFrame,
    subset_meta: Dict[str, object],
    global_explanations: Dict[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "df": df,
        "subset_meta": subset_meta,
        "global_explanations": global_explanations,
    }
    pd.to_pickle(payload, cache_path)


def _protocol_targets(total_rows: int, protocols: Iterable[str]) -> Dict[str, int]:
    names = list(protocols)
    if not names:
        return {}
    base = int(total_rows) // len(names)
    rem = int(total_rows) % len(names)
    out = {}
    for i, proto in enumerate(names):
        out[proto] = base + (1 if i < rem else 0)
    return out


def _score_quantiles(values: pd.Series) -> Dict[str, float]:
    if values.empty:
        return {"q10": 0.0, "q50": 0.0, "q90": 0.0}
    q = values.quantile([0.1, 0.5, 0.9]).to_dict()
    return {
        "q10": safe_float(q.get(0.1), 0.0),
        "q50": safe_float(q.get(0.5), 0.0),
        "q90": safe_float(q.get(0.9), 0.0),
    }


def _move_dynamic_alerts_to_front(
    df: pd.DataFrame,
    shuffle_seed: int,
    prefix_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if df.empty:
        return df, {"requested_prefix_rows": int(prefix_rows), "actual_prefix_rows": 0}

    work = df.copy()
    if "margin" not in work.columns:
        work["margin"] = pd.to_numeric(work.get("score_attack", 0.0), errors="coerce").fillna(0.0) - pd.to_numeric(
            work.get("threshold", 0.5), errors="coerce"
        ).fillna(0.5)
    work["abs_margin"] = work["margin"].abs()
    work["_row_order"] = np.arange(len(work), dtype=int)

    score = pd.to_numeric(work.get("score_attack", 0.0), errors="coerce").fillna(0.0)
    pred = pd.to_numeric(work.get("prediction", 0), errors="coerce").fillna(0).astype(np.int8)
    proto = work.get("protocol_hint", pd.Series([""] * len(work), index=work.index)).astype(str)
    dyn_mask = (pred == 1) & (score >= pd.to_numeric(work.get("threshold", 0.5), errors="coerce").fillna(0.5)) & (score < 0.9999)
    if not bool(dyn_mask.any()):
        return work.drop(columns=["_row_order"], errors="ignore"), {
            "requested_prefix_rows": int(prefix_rows),
            "actual_prefix_rows": 0,
            "note": "no_non_saturated_alerts_available",
        }

    requested = max(0, int(prefix_rows))
    selected_idx: List[int] = []
    selected_counts: Dict[str, int] = {}
    per_proto = max(1, requested // max(1, len(PROTOCOLS)))

    for p in PROTOCOLS:
        sub = work[(proto == p) & dyn_mask].sort_values(["abs_margin", "score_attack", "_row_order"], ascending=True)
        take_n = min(per_proto, len(sub))
        if take_n > 0:
            picked = sub.head(take_n).index.to_list()
            selected_idx.extend(picked)
            selected_counts[p] = int(len(picked))

    if len(selected_idx) < requested:
        extra = work[dyn_mask & ~work.index.isin(selected_idx)].sort_values(
            ["abs_margin", "score_attack", "_row_order"], ascending=True
        )
        need = requested - len(selected_idx)
        selected_idx.extend(extra.head(need).index.to_list())

    if not selected_idx:
        return work.drop(columns=["_row_order"], errors="ignore"), {
            "requested_prefix_rows": int(prefix_rows),
            "actual_prefix_rows": 0,
        }

    prefix_df = work.loc[selected_idx].copy().reset_index(drop=True)
    prefix_df = interleave_by_protocol_source(prefix_df, shuffle_seed=int(shuffle_seed))
    remainder = work.drop(index=selected_idx).reset_index(drop=True)
    out = pd.concat([prefix_df, remainder], ignore_index=True).drop(columns=["_row_order"], errors="ignore")

    return out, {
        "requested_prefix_rows": int(prefix_rows),
        "actual_prefix_rows": int(len(prefix_df)),
        "per_protocol_selected": {str(k): int(v) for k, v in sorted(selected_counts.items())},
    }


def _sample_strata_one_pass(
    csv_paths: List[Path],
    usecols: List[str],
    protocol_targets: Dict[str, int],
    seed: int,
    chunk_size: int,
) -> Tuple[Dict[Tuple[str, int], pd.DataFrame], Dict[Tuple[str, int], int]]:
    rng = np.random.default_rng(int(seed))
    samples: Dict[Tuple[str, int], pd.DataFrame] = {}
    available_counts: Dict[Tuple[str, int], int] = {}
    for proto in PROTOCOLS:
        for label in (0, 1):
            samples[(proto, label)] = pd.DataFrame(columns=usecols + ["_rand"])
            available_counts[(proto, label)] = 0

    for csv_path in csv_paths:
        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
        for chunk in reader:
            proto = normalize_protocol_series(chunk["protocol_hint"])
            label = normalize_label_series(chunk["label"])
            chunk = chunk.copy()
            chunk["protocol_hint"] = proto
            chunk["label"] = label

            for p in PROTOCOLS:
                proto_mask = proto == p
                if not bool(proto_mask.any()):
                    continue
                keep_n = int(protocol_targets[p])
                for l in (0, 1):
                    key = (p, l)
                    mask = proto_mask & (label == l)
                    n = int(mask.sum())
                    if n <= 0:
                        continue

                    available_counts[key] += n
                    sub = chunk.loc[mask, usecols].copy()
                    sub["_rand"] = rng.random(n)
                    merged = pd.concat([samples[key], sub], ignore_index=True)
                    if len(merged) > keep_n:
                        merged = merged.nsmallest(keep_n, "_rand").reset_index(drop=True)
                    samples[key] = merged

    return samples, available_counts


def build_in_memory_demo_subset(
    train_csv: Path,
    test_csv: Path,
    feature_columns: List[str],
    total_rows: int,
    attack_ratio: float,
    seed: int,
    replay_order: str,
    shuffle_seed: int,
    chunk_size: int = 200000,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    train_cols = read_csv_columns(train_csv)
    test_cols = read_csv_columns(test_csv)
    shared = set(train_cols).intersection(test_cols)

    metadata_cols = ["label", "protocol_hint", "attack_family", "source_relpath", "source_row_index"]
    usecols = [c for c in metadata_cols if c in shared] + [c for c in feature_columns if c in shared]

    if "protocol_hint" not in usecols or "label" not in usecols:
        raise RuntimeError("train/test CSV must include protocol_hint and label for subset sampling")

    protocol_targets = _protocol_targets(total_rows=total_rows, protocols=PROTOCOLS)
    per_protocol_desired: Dict[str, Dict[str, int]] = {}
    for proto in PROTOCOLS:
        proto_total = int(protocol_targets[proto])
        attack_target = int(round(proto_total * float(attack_ratio)))
        attack_target = max(0, min(proto_total, attack_target))
        benign_target = proto_total - attack_target
        per_protocol_desired[proto] = {
            "target_total": proto_total,
            "target_attack": attack_target,
            "target_benign": benign_target,
        }

    csv_paths = [train_csv, test_csv]
    sampled_by_stratum, available_counts = _sample_strata_one_pass(
        csv_paths=csv_paths,
        usecols=usecols,
        protocol_targets=protocol_targets,
        seed=seed,
        chunk_size=chunk_size,
    )

    per_protocol_actual: Dict[str, Dict[str, int]] = {}
    per_protocol_fallback: Dict[str, Dict[str, int]] = {}
    final_rows: List[dict] = []
    rng = np.random.default_rng(int(seed) + 99)

    for proto in PROTOCOLS:
        desired = per_protocol_desired[proto]
        target_total = int(desired["target_total"])
        target_attack = int(desired["target_attack"])
        target_benign = int(desired["target_benign"])

        pos_rows = sampled_by_stratum[(proto, 1)].drop(columns=["_rand"], errors="ignore").to_dict(orient="records")
        neg_rows = sampled_by_stratum[(proto, 0)].drop(columns=["_rand"], errors="ignore").to_dict(orient="records")

        rng.shuffle(pos_rows)
        rng.shuffle(neg_rows)

        take_pos = min(len(pos_rows), target_attack)
        take_neg = min(len(neg_rows), target_benign)

        chosen_pos = pos_rows[:take_pos]
        chosen_neg = neg_rows[:take_neg]

        pos_extra = pos_rows[take_pos:]
        neg_extra = neg_rows[take_neg:]

        fallback_from_benign_for_attack = 0
        fallback_from_attack_for_benign = 0

        missing_attack = max(0, target_attack - take_pos)
        if missing_attack > 0 and neg_extra:
            n = min(missing_attack, len(neg_extra))
            chosen_neg.extend(neg_extra[:n])
            neg_extra = neg_extra[n:]
            fallback_from_benign_for_attack = n

        missing_benign = max(0, target_benign - take_neg)
        if missing_benign > 0 and pos_extra:
            n = min(missing_benign, len(pos_extra))
            chosen_pos.extend(pos_extra[:n])
            pos_extra = pos_extra[n:]
            fallback_from_attack_for_benign = n

        proto_rows = chosen_pos + chosen_neg
        if len(proto_rows) < target_total:
            remaining = target_total - len(proto_rows)
            fill_pool = pos_extra + neg_extra
            if fill_pool:
                rng.shuffle(fill_pool)
                proto_rows.extend(fill_pool[:remaining])

        proto_rows = proto_rows[:target_total]
        final_rows.extend(proto_rows)

        actual_attack = sum(safe_int(r.get("label"), 0) == 1 for r in proto_rows)
        actual_benign = len(proto_rows) - actual_attack

        per_protocol_actual[proto] = {
            "rows": int(len(proto_rows)),
            "attack": int(actual_attack),
            "benign": int(actual_benign),
        }
        per_protocol_fallback[proto] = {
            "from_benign_for_attack_deficit": int(fallback_from_benign_for_attack),
            "from_attack_for_benign_deficit": int(fallback_from_attack_for_benign),
        }

    if not final_rows:
        raise RuntimeError("Failed to sample replay subset. No rows selected.")

    demo_df = pd.DataFrame(final_rows)
    if "protocol_hint" not in demo_df.columns:
        demo_df["protocol_hint"] = "unknown"

    demo_df["protocol_hint"] = normalize_protocol_series(demo_df["protocol_hint"])
    demo_df["label"] = normalize_label_series(demo_df.get("label", 0))

    demo_df = reorder_rows(demo_df, replay_order=replay_order, shuffle_seed=shuffle_seed)

    replay_protocol_counts = demo_df["protocol_hint"].astype(str).value_counts().to_dict()
    replay_label_counts = demo_df["label"].astype(int).value_counts().to_dict()

    meta = {
        "seed": int(seed),
        "total_rows_requested": int(total_rows),
        "total_rows_selected": int(len(demo_df)),
        "attack_ratio_requested": float(attack_ratio),
        "protocol_targets": {k: int(v) for k, v in protocol_targets.items()},
        "desired_by_protocol": per_protocol_desired,
        "available_by_stratum": {
            f"{proto}|{label}": int(available_counts.get((proto, label), 0))
            for proto in PROTOCOLS
            for label in (0, 1)
        },
        "actual_by_protocol": per_protocol_actual,
        "fallback_by_protocol": per_protocol_fallback,
        "replay_protocol_counts": {str(k): int(v) for k, v in replay_protocol_counts.items()},
        "replay_label_counts": {str(k): int(v) for k, v in replay_label_counts.items()},
        "source_paths": [str(train_csv), str(test_csv)],
    }
    return demo_df.reset_index(drop=True), meta


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    if df.empty:
        return np.empty((0, len(feature_columns)), dtype=np.float32)
    x_df = df.reindex(columns=feature_columns, fill_value=0.0).copy()
    for col in feature_columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce").fillna(0.0)
    return x_df.to_numpy(dtype=np.float32, copy=False)


def infer_protocol_labels(df: pd.DataFrame, default_protocol: str) -> np.ndarray:
    if "protocol_hint" not in df.columns:
        return np.array([default_protocol] * len(df), dtype=object)
    raw = normalize_protocol_series(df["protocol_hint"])
    arr = raw.to_numpy(dtype=object, copy=False)
    out = np.array([v if v in PROTOCOLS else default_protocol for v in arr], dtype=object)
    return out


def routed_predict_catboost(
    df: pd.DataFrame,
    feature_columns: List[str],
    models_by_protocol: Dict[str, CatBoostClassifier],
    thresholds_by_protocol: Dict[str, float],
    prediction_task_type: str = "CPU",
    default_protocol: str = "wifi",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["protocol_used", "score_attack", "threshold", "prediction"])

    routed_protocol = infer_protocol_labels(df, default_protocol=default_protocol)
    x_mat = prepare_feature_matrix(df, feature_columns)

    score = np.zeros(len(df), dtype=np.float32)
    thr_out = np.zeros(len(df), dtype=np.float32)
    pred = np.zeros(len(df), dtype=np.int8)

    for proto in sorted(set(routed_protocol.tolist())):
        mask = routed_protocol == proto
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        model_proto = proto if proto in models_by_protocol else default_protocol
        model = models_by_protocol[model_proto]
        thr = float(thresholds_by_protocol.get(model_proto, 0.5))
        pool = Pool(x_mat[idx])
        p = model.predict_proba(pool, task_type=str(prediction_task_type).upper())[:, 1].astype(np.float32, copy=False)

        score[idx] = p
        thr_out[idx] = thr
        pred[idx] = (p >= thr).astype(np.int8, copy=False)
        routed_protocol[idx] = model_proto

    return pd.DataFrame(
        {
            "protocol_used": routed_protocol,
            "score_attack": score,
            "threshold": thr_out,
            "prediction": pred,
        }
    )


def build_in_memory_demo_subset_full_sweep(
    train_csv: Path,
    test_csv: Path,
    feature_columns: List[str],
    models_by_protocol: Dict[str, CatBoostClassifier],
    thresholds_by_protocol: Dict[str, float],
    total_rows: int,
    attack_ratio: float,
    seed: int,
    replay_order: str,
    shuffle_seed: int,
    prediction_task_type: str,
    chunk_size: int = 150000,
    dynamic_fraction: float = 0.72,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    train_cols = read_csv_columns(train_csv)
    test_cols = read_csv_columns(test_csv)
    shared = set(train_cols).intersection(test_cols)

    metadata_cols = ["label", "protocol_hint", "attack_family", "source_relpath", "source_row_index"]
    usecols = [c for c in metadata_cols if c in shared] + [c for c in feature_columns if c in shared]
    if "protocol_hint" not in usecols or "label" not in usecols:
        raise RuntimeError("train/test CSV must include protocol_hint and label for subset sampling")

    protocol_targets = _protocol_targets(total_rows=total_rows, protocols=PROTOCOLS)
    per_protocol_desired: Dict[str, Dict[str, int]] = {}
    stratum_pool_sizes: Dict[Tuple[str, int], Dict[str, int]] = {}
    for proto in PROTOCOLS:
        proto_total = int(protocol_targets[proto])
        attack_target = int(round(proto_total * float(attack_ratio)))
        attack_target = max(0, min(proto_total, attack_target))
        benign_target = proto_total - attack_target
        per_protocol_desired[proto] = {
            "target_total": proto_total,
            "target_attack": attack_target,
            "target_benign": benign_target,
        }
        for lbl, target in ((1, attack_target), (0, benign_target)):
            keep_total = max(int(target) + 700, int(math.ceil(float(target) * 2.2)))
            keep_dynamic = max(int(target), int(round(float(keep_total) * float(dynamic_fraction))))
            keep_random = max(0, int(keep_total - keep_dynamic))
            stratum_pool_sizes[(proto, lbl)] = {
                "target": int(target),
                "keep_total": int(keep_total),
                "keep_dynamic": int(keep_dynamic),
                "keep_random": int(keep_random),
            }

    scored_cols = ["protocol_used", "score_attack", "threshold", "prediction", "margin", "abs_margin"]
    keep_cols = usecols + [c for c in scored_cols if c not in usecols]
    dynamic_samples: Dict[Tuple[str, int], pd.DataFrame] = {}
    random_samples: Dict[Tuple[str, int], pd.DataFrame] = {}
    available_counts: Dict[Tuple[str, int], int] = {}
    for proto in PROTOCOLS:
        for lbl in (0, 1):
            dynamic_samples[(proto, lbl)] = pd.DataFrame(columns=keep_cols + ["_priority"])
            random_samples[(proto, lbl)] = pd.DataFrame(columns=keep_cols + ["_rand"])
            available_counts[(proto, lbl)] = 0

    sweep_rows_by_protocol = defaultdict(int)
    sweep_alerts_by_protocol = defaultdict(int)
    sweep_non_sat_alerts_by_protocol = defaultdict(int)
    sweep_total_rows = 0

    rng = np.random.default_rng(int(seed))
    csv_paths = [train_csv, test_csv]
    for csv_path in csv_paths:
        reader = pd.read_csv(csv_path, usecols=usecols, chunksize=int(chunk_size))
        for chunk in reader:
            chunk = chunk.copy()
            chunk["protocol_hint"] = normalize_protocol_series(chunk["protocol_hint"])
            chunk["label"] = normalize_label_series(chunk["label"])
            chunk = chunk[chunk["protocol_hint"].astype(str).isin(PROTOCOLS)].reset_index(drop=True)
            if chunk.empty:
                continue

            pred = routed_predict_catboost(
                df=chunk,
                feature_columns=feature_columns,
                models_by_protocol=models_by_protocol,
                thresholds_by_protocol=thresholds_by_protocol,
                prediction_task_type=prediction_task_type,
                default_protocol="wifi",
            )
            chunk = pd.concat([chunk, pred], axis=1)
            chunk["score_attack"] = pd.to_numeric(chunk["score_attack"], errors="coerce").fillna(0.0)
            chunk["threshold"] = pd.to_numeric(chunk["threshold"], errors="coerce").fillna(0.5)
            chunk["prediction"] = pd.to_numeric(chunk["prediction"], errors="coerce").fillna(0).astype(np.int8)
            chunk["margin"] = chunk["score_attack"] - chunk["threshold"]
            chunk["abs_margin"] = chunk["margin"].abs()

            sweep_total_rows += int(len(chunk))
            for proto in PROTOCOLS:
                p_mask = chunk["protocol_hint"].astype(str) == proto
                p_n = int(p_mask.sum())
                if p_n <= 0:
                    continue
                sweep_rows_by_protocol[proto] += p_n
                alert_mask = p_mask & (chunk["prediction"] == 1)
                a_n = int(alert_mask.sum())
                sweep_alerts_by_protocol[proto] += a_n
                if a_n > 0:
                    non_sat = int((alert_mask & (chunk["score_attack"] < 0.9999)).sum())
                    sweep_non_sat_alerts_by_protocol[proto] += non_sat

            for proto in PROTOCOLS:
                proto_mask = chunk["protocol_hint"].astype(str) == proto
                if not bool(proto_mask.any()):
                    continue
                for lbl in (0, 1):
                    key = (proto, lbl)
                    mask = proto_mask & (chunk["label"] == lbl)
                    n = int(mask.sum())
                    if n <= 0:
                        continue
                    available_counts[key] += n
                    sub = chunk.loc[mask, keep_cols].copy()
                    keep_dynamic = int(stratum_pool_sizes[key]["keep_dynamic"])
                    keep_random = int(stratum_pool_sizes[key]["keep_random"])

                    if keep_dynamic > 0:
                        dyn = sub.copy()
                        dyn["_priority"] = pd.to_numeric(dyn["abs_margin"], errors="coerce").fillna(999.0)
                        merged = pd.concat([dynamic_samples[key], dyn], ignore_index=True)
                        if len(merged) > keep_dynamic:
                            merged = merged.nsmallest(keep_dynamic, "_priority")
                        dynamic_samples[key] = merged.reset_index(drop=True)

                    if keep_random > 0:
                        rnd = sub.copy()
                        rnd["_rand"] = rng.random(n)
                        merged = pd.concat([random_samples[key], rnd], ignore_index=True)
                        if len(merged) > keep_random:
                            merged = merged.nsmallest(keep_random, "_rand")
                        random_samples[key] = merged.reset_index(drop=True)

    sampled_by_stratum: Dict[Tuple[str, int], pd.DataFrame] = {}
    id_cols = [c for c in ("source_relpath", "source_row_index", "protocol_hint", "label") if c in keep_cols]
    for proto in PROTOCOLS:
        for lbl in (0, 1):
            key = (proto, lbl)
            dyn = dynamic_samples[key].drop(columns=["_priority"], errors="ignore")
            rnd = random_samples[key].drop(columns=["_rand"], errors="ignore")
            merged = pd.concat([dyn, rnd], ignore_index=True)
            if merged.empty:
                sampled_by_stratum[key] = merged
                continue
            if id_cols:
                merged = merged.drop_duplicates(subset=id_cols, keep="first")
            else:
                merged = merged.drop_duplicates()
            merged = merged.sort_values(["abs_margin", "score_attack"], ascending=[True, True]).reset_index(drop=True)
            sampled_by_stratum[key] = merged

    per_protocol_actual: Dict[str, Dict[str, int]] = {}
    per_protocol_fallback: Dict[str, Dict[str, int]] = {}
    final_parts: List[pd.DataFrame] = []

    for proto in PROTOCOLS:
        desired = per_protocol_desired[proto]
        target_total = int(desired["target_total"])
        target_attack = int(desired["target_attack"])
        target_benign = int(desired["target_benign"])

        pos = sampled_by_stratum[(proto, 1)].copy()
        neg = sampled_by_stratum[(proto, 0)].copy()

        take_pos = min(len(pos), target_attack)
        take_neg = min(len(neg), target_benign)
        chosen_pos = pos.head(take_pos).copy()
        chosen_neg = neg.head(take_neg).copy()
        pos_extra = pos.iloc[take_pos:].copy()
        neg_extra = neg.iloc[take_neg:].copy()

        fallback_from_benign_for_attack = 0
        fallback_from_attack_for_benign = 0

        missing_attack = max(0, target_attack - len(chosen_pos))
        if missing_attack > 0 and not neg_extra.empty:
            add = neg_extra.head(missing_attack).copy()
            chosen_neg = pd.concat([chosen_neg, add], ignore_index=True)
            neg_extra = neg_extra.iloc[len(add) :].copy()
            fallback_from_benign_for_attack = int(len(add))

        missing_benign = max(0, target_benign - len(chosen_neg))
        if missing_benign > 0 and not pos_extra.empty:
            add = pos_extra.head(missing_benign).copy()
            chosen_pos = pd.concat([chosen_pos, add], ignore_index=True)
            pos_extra = pos_extra.iloc[len(add) :].copy()
            fallback_from_attack_for_benign = int(len(add))

        proto_df = pd.concat([chosen_pos, chosen_neg], ignore_index=True)
        if len(proto_df) < target_total:
            fill_pool = pd.concat([pos_extra, neg_extra], ignore_index=True)
            if not fill_pool.empty:
                fill_pool = fill_pool.sort_values(["abs_margin", "score_attack"], ascending=[True, True])
                need = target_total - len(proto_df)
                proto_df = pd.concat([proto_df, fill_pool.head(need)], ignore_index=True)

        proto_df = proto_df.head(target_total).reset_index(drop=True)
        final_parts.append(proto_df)

        actual_attack = int((pd.to_numeric(proto_df.get("label", 0), errors="coerce").fillna(0).astype(np.int8) == 1).sum())
        actual_benign = int(len(proto_df) - actual_attack)
        per_protocol_actual[proto] = {
            "rows": int(len(proto_df)),
            "attack": int(actual_attack),
            "benign": int(actual_benign),
        }
        per_protocol_fallback[proto] = {
            "from_benign_for_attack_deficit": int(fallback_from_benign_for_attack),
            "from_attack_for_benign_deficit": int(fallback_from_attack_for_benign),
        }

    demo_df = pd.concat(final_parts, ignore_index=True) if final_parts else pd.DataFrame(columns=keep_cols)
    if demo_df.empty:
        raise RuntimeError("Failed to sample replay subset from full sweep. No rows selected.")

    demo_df["protocol_hint"] = normalize_protocol_series(demo_df["protocol_hint"])
    demo_df["label"] = normalize_label_series(demo_df["label"])
    demo_df["score_attack"] = pd.to_numeric(demo_df.get("score_attack", 0.0), errors="coerce").fillna(0.0)
    demo_df["threshold"] = pd.to_numeric(demo_df.get("threshold", 0.5), errors="coerce").fillna(0.5)
    demo_df["prediction"] = pd.to_numeric(demo_df.get("prediction", 0), errors="coerce").fillna(0).astype(np.int8)
    demo_df["margin"] = demo_df["score_attack"] - demo_df["threshold"]
    demo_df["abs_margin"] = demo_df["margin"].abs()

    demo_df = reorder_rows(demo_df, replay_order=replay_order, shuffle_seed=shuffle_seed)
    prefix_rows = max(len(PROTOCOLS) * 60, int(round(len(demo_df) * 0.12)))
    demo_df, dynamic_front_meta = _move_dynamic_alerts_to_front(
        demo_df,
        shuffle_seed=int(shuffle_seed),
        prefix_rows=int(prefix_rows),
    )

    replay_protocol_counts = demo_df["protocol_hint"].astype(str).value_counts().to_dict()
    replay_label_counts = demo_df["label"].astype(int).value_counts().to_dict()
    replay_alerts_by_protocol = {}
    replay_non_sat_alerts_by_protocol = {}
    replay_alert_score_quantiles = {}
    for proto in PROTOCOLS:
        proto_mask = demo_df["protocol_hint"].astype(str) == proto
        alerts = demo_df[proto_mask & (demo_df["prediction"].astype(np.int8) == 1)]["score_attack"]
        replay_alerts_by_protocol[proto] = int(len(alerts))
        replay_non_sat_alerts_by_protocol[proto] = int((alerts < 0.9999).sum())
        replay_alert_score_quantiles[proto] = _score_quantiles(alerts)

    meta = {
        "seed": int(seed),
        "sampling_mode": "full_dataset_score_sweep",
        "total_rows_requested": int(total_rows),
        "total_rows_selected": int(len(demo_df)),
        "attack_ratio_requested": float(attack_ratio),
        "protocol_targets": {k: int(v) for k, v in protocol_targets.items()},
        "desired_by_protocol": per_protocol_desired,
        "pool_sizes_by_stratum": {
            f"{proto}|{lbl}": {
                "target": int(v["target"]),
                "keep_total": int(v["keep_total"]),
                "keep_dynamic": int(v["keep_dynamic"]),
                "keep_random": int(v["keep_random"]),
            }
            for (proto, lbl), v in stratum_pool_sizes.items()
        },
        "available_by_stratum": {
            f"{proto}|{label}": int(available_counts.get((proto, label), 0))
            for proto in PROTOCOLS
            for label in (0, 1)
        },
        "actual_by_protocol": per_protocol_actual,
        "fallback_by_protocol": per_protocol_fallback,
        "replay_protocol_counts": {str(k): int(v) for k, v in replay_protocol_counts.items()},
        "replay_label_counts": {str(k): int(v) for k, v in replay_label_counts.items()},
        "replay_alerts_by_protocol": {str(k): int(v) for k, v in replay_alerts_by_protocol.items()},
        "replay_non_saturated_alerts_by_protocol": {str(k): int(v) for k, v in replay_non_sat_alerts_by_protocol.items()},
        "replay_alert_score_quantiles": {
            str(k): {"q10": safe_float(v["q10"]), "q50": safe_float(v["q50"]), "q90": safe_float(v["q90"])}
            for k, v in replay_alert_score_quantiles.items()
        },
        "dynamic_front": dynamic_front_meta,
        "sweep": {
            "rows_scored_total": int(sweep_total_rows),
            "rows_scored_by_protocol": {str(k): int(v) for k, v in sweep_rows_by_protocol.items()},
            "alerts_by_protocol": {str(k): int(v) for k, v in sweep_alerts_by_protocol.items()},
            "non_saturated_alerts_by_protocol": {
                str(k): int(v) for k, v in sweep_non_sat_alerts_by_protocol.items()
            },
        },
        "source_paths": [str(train_csv), str(test_csv)],
    }
    return demo_df.reset_index(drop=True), meta


def local_feature_contributions_catboost(
    model: CatBoostClassifier,
    row_features: np.ndarray,
    feature_columns: List[str],
    top_n: int,
) -> pd.DataFrame:
    row = np.asarray(row_features, dtype=np.float32).reshape(1, -1)
    shap = model.get_feature_importance(data=Pool(row), type="ShapValues")
    values = shap[0, :-1]
    bias = float(shap[0, -1])

    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "contribution": values,
        }
    )
    df["abs_contribution"] = df["contribution"].abs()
    df["direction"] = np.where(df["contribution"] >= 0.0, "attack", "benign")
    df = df.sort_values("abs_contribution", ascending=False).head(max(1, int(top_n))).reset_index(drop=True)
    df.insert(0, "bias", bias)
    return df


def build_global_shap_payload(
    df: pd.DataFrame,
    feature_columns: List[str],
    models_by_protocol: Dict[str, CatBoostClassifier],
    thresholds_by_protocol: Dict[str, float],
) -> Dict[str, object]:
    protocol_payload = []
    overall_scores = defaultdict(float)

    for proto in PROTOCOLS:
        sub = df[df["protocol_hint"].astype(str) == proto].copy()
        if sub.empty:
            protocol_payload.append(
                {
                    "protocol": proto,
                    "threshold": safe_float(thresholds_by_protocol.get(proto, 0.5), 0.5),
                    "top_features": [],
                }
            )
            continue

        x = prepare_feature_matrix(sub, feature_columns)
        shap = models_by_protocol[proto].get_feature_importance(data=Pool(x), type="ShapValues")[:, :-1]
        mean_abs = np.abs(shap).mean(axis=0)

        proto_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_contribution": mean_abs.astype(np.float64, copy=False),
            }
        ).sort_values("mean_abs_contribution", ascending=False)

        top_features = []
        for _, row in proto_df.head(20).iterrows():
            fname = safe_str(row.get("feature"))
            score = safe_float(row.get("mean_abs_contribution"), 0.0)
            top_features.append(
                {
                    "feature": fname,
                    "mean_abs_contribution": score,
                }
            )
            overall_scores[fname] += score

        protocol_payload.append(
            {
                "protocol": proto,
                "threshold": safe_float(thresholds_by_protocol.get(proto, 0.5), 0.5),
                "top_features": top_features,
            }
        )

    overall_top = [
        {"feature": fname, "score": float(score)}
        for fname, score in sorted(overall_scores.items(), key=lambda kv: kv[1], reverse=True)[:25]
    ]
    return {
        "protocols": protocol_payload,
        "overall_top_features": overall_top,
    }


class LiveIDSRuntime:
    def __init__(
        self,
        base_run_dir: Path,
        matrix_run_dir: Path,
        train_csv: Path,
        test_csv: Path,
        demo_cache_pickle: Path | None,
        threshold_policy: str,
        inference_device: str,
        rows_per_second: int,
        local_top_n: int,
        replay_order: str,
        shuffle_seed: int,
        max_recent_alerts: int,
        demo_total_rows: int,
        demo_attack_ratio: float,
        demo_seed: int,
        demo_cache_mode: str,
        demo_cache_dir: Path,
    ) -> None:
        self.base_run_dir = base_run_dir
        self.matrix_run_dir = matrix_run_dir
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.demo_cache_pickle = demo_cache_pickle
        self.threshold_policy = threshold_policy
        self.inference_device_requested = safe_str(inference_device).strip().lower() or "auto"
        self.rows_per_second = max(1, min(10, int(rows_per_second)))
        self.local_top_n = max(1, int(local_top_n))
        self.replay_order = replay_order
        self.shuffle_seed = int(shuffle_seed)
        self.max_recent_alerts = max(1, int(max_recent_alerts))
        self.demo_total_rows = max(3, int(demo_total_rows))
        self.demo_attack_ratio = max(0.0, min(1.0, float(demo_attack_ratio)))
        self.demo_seed = int(demo_seed)
        self.demo_cache_mode = str(demo_cache_mode)
        self.demo_cache_dir = demo_cache_dir

        self.lock = threading.Lock()

        self.feature_columns = load_feature_columns(base_run_dir)
        self.models_by_protocol = load_catboost_e_models_by_protocol(matrix_run_dir)
        self.thresholds_by_protocol = load_thresholds_by_policy(matrix_run_dir, policy=threshold_policy)
        self.prediction_task_type, self.inference_device_note = resolve_prediction_task_type(
            models_by_protocol=self.models_by_protocol,
            feature_count=len(self.feature_columns),
            inference_device=self.inference_device_requested,
        )
        if self.prediction_task_type.upper() == "GPU":
            self.shap_models_by_protocol = load_catboost_e_models_by_protocol(matrix_run_dir)
        else:
            self.shap_models_by_protocol = self.models_by_protocol

        cache_hit = False
        cache_error = ""
        cache_source = "generated"
        loaded = None
        if self.demo_cache_pickle is not None:
            self.demo_cache_path = self.demo_cache_pickle
            cache_source = "bundled_pickle"
            loaded = load_demo_cache(self.demo_cache_path)
            if loaded is None:
                raise RuntimeError(f"Could not load bundled demo cache: {self.demo_cache_path}")
            cache_hit = True
        else:
            self.demo_cache_path = build_demo_cache_path(
                cache_dir=self.demo_cache_dir,
                train_csv=self.train_csv,
                test_csv=self.test_csv,
                feature_columns=self.feature_columns,
                total_rows=self.demo_total_rows,
                attack_ratio=self.demo_attack_ratio,
                seed=self.demo_seed,
                replay_order=self.replay_order,
                shuffle_seed=self.shuffle_seed,
            )
            if self.demo_cache_mode == "reuse":
                loaded = load_demo_cache(self.demo_cache_path)
                if loaded is not None:
                    cache_hit = True
                    cache_source = "generated_reuse"
                elif self.demo_cache_path.exists():
                    cache_error = "cache_present_but_invalid"

        if loaded is not None:
            full_df, subset_meta, global_explanations = loaded
        else:
            full_df, subset_meta = build_in_memory_demo_subset_full_sweep(
                train_csv=self.train_csv,
                test_csv=self.test_csv,
                feature_columns=self.feature_columns,
                models_by_protocol=self.models_by_protocol,
                thresholds_by_protocol=self.thresholds_by_protocol,
                total_rows=self.demo_total_rows,
                attack_ratio=self.demo_attack_ratio,
                seed=self.demo_seed,
                replay_order=self.replay_order,
                shuffle_seed=self.shuffle_seed,
                prediction_task_type=self.prediction_task_type,
            )
            global_explanations = build_global_shap_payload(
                df=full_df,
                feature_columns=self.feature_columns,
                models_by_protocol=self.shap_models_by_protocol,
                thresholds_by_protocol=self.thresholds_by_protocol,
            )
            if self.demo_cache_mode != "off":
                try:
                    save_demo_cache(
                        cache_path=self.demo_cache_path,
                        df=full_df,
                        subset_meta=subset_meta,
                        global_explanations=global_explanations,
                    )
                except Exception:
                    cache_error = "cache_write_failed"

        subset_meta = dict(subset_meta)
        subset_meta["cache"] = {
            "mode": self.demo_cache_mode,
            "hit": bool(cache_hit),
            "path": str(self.demo_cache_path),
            "error": str(cache_error),
            "source": str(cache_source),
        }

        self.df = full_df.reset_index(drop=True)
        self.total_rows = int(len(self.df))
        self.subset_meta = subset_meta
        self.global_explanations = global_explanations

        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.ended = False
        self.cursor = 0
        self.sim_seconds = 0
        self.flows_processed = 0
        self.alerts_detected = 0
        self.alerts_surfaced = 0
        self.first_alert_sim_second = None
        self.first_alert_row = None
        self.recent_alerts: List[Dict[str, object]] = []
        self.protocol_flow_counts = defaultdict(int)
        self.protocol_alert_counts = defaultdict(int)
        self.last_update = time.monotonic()

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def set_rows_per_second(self, rows_per_second: int) -> int:
        with self.lock:
            self._advance_locked()
            self.rows_per_second = max(1, min(10, int(rows_per_second)))
            self.last_update = time.monotonic()
            return int(self.rows_per_second)

    def start(self) -> None:
        with self.lock:
            if not self.ended:
                self.running = True
                self.last_update = time.monotonic()

    def pause(self) -> None:
        with self.lock:
            self._advance_locked()
            self.running = False

    def jump_to_first_alert(self) -> None:
        with self.lock:
            if self.first_alert_sim_second is not None:
                self.running = False
                self.last_update = time.monotonic()
                return

            self.running = True
            while not self.ended and self.first_alert_sim_second is None:
                self._advance_one_second_locked()
            self.running = False
            self.last_update = time.monotonic()

    def _build_alert_payload(
        self,
        row: pd.Series,
        x_row: np.ndarray,
        row_index: int,
    ) -> Dict[str, object]:
        proto = safe_str(row.get("protocol_used"))
        model = self.shap_models_by_protocol.get(proto)
        local_features = []
        if model is not None:
            local_df = local_feature_contributions_catboost(
                model=model,
                row_features=x_row,
                feature_columns=self.feature_columns,
                top_n=self.local_top_n,
            )
            for _, lr in local_df.iterrows():
                local_features.append(
                    {
                        "feature": safe_str(lr.get("feature")),
                        "contribution": safe_float(lr.get("contribution")),
                        "abs_contribution": safe_float(lr.get("abs_contribution")),
                        "direction": safe_str(lr.get("direction")),
                    }
                )

        return {
            "id": f"row_{row_index:07d}",
            "global_row_index": row_index,
            "protocol": proto,
            "score_attack": safe_float(row.get("score_attack")),
            "threshold": safe_float(row.get("threshold"), 0.5),
            "attack_family": safe_str(row.get("attack_family")),
            "source_relpath": safe_str(row.get("source_relpath")),
            "source_row_index": safe_int(row.get("source_row_index")),
            "sim_second": int(self.sim_seconds),
            "local_explanation": local_features,
        }

    def _advance_one_second_locked(self) -> None:
        if self.ended:
            return

        start = self.cursor
        end = min(self.total_rows, start + self.rows_per_second)
        if start >= end:
            self.ended = True
            self.running = False
            return

        chunk = self.df.iloc[start:end].copy().reset_index(drop=True)
        chunk = chunk.drop(
            columns=["protocol_used", "score_attack", "threshold", "prediction", "margin", "abs_margin"],
            errors="ignore",
        )
        x_chunk = prepare_feature_matrix(chunk, self.feature_columns)
        pred = routed_predict_catboost(
            df=chunk,
            feature_columns=self.feature_columns,
            models_by_protocol=self.models_by_protocol,
            thresholds_by_protocol=self.thresholds_by_protocol,
            prediction_task_type=self.prediction_task_type,
            default_protocol="wifi",
        )
        scored = pd.concat([chunk, pred], axis=1)

        second_alerts: List[Dict[str, object]] = []
        for i in range(len(scored)):
            row = scored.iloc[i]
            proto = safe_str(row.get("protocol_used"))
            self.protocol_flow_counts[proto] += 1
            if safe_int(row.get("prediction")) == 1:
                self.alerts_detected += 1
                self.alerts_surfaced += 1
                self.protocol_alert_counts[proto] += 1
                row_index = start + i
                if self.first_alert_sim_second is None:
                    self.first_alert_sim_second = int(self.sim_seconds + 1)
                    self.first_alert_row = int(row_index)
                second_alerts.append(self._build_alert_payload(row=row, x_row=x_chunk[i], row_index=row_index))

        if second_alerts:
            self.recent_alerts = (second_alerts + self.recent_alerts)[: self.max_recent_alerts]

        self.cursor = end
        self.flows_processed = end
        self.sim_seconds += 1
        if self.cursor >= self.total_rows:
            self.ended = True
            self.running = False

    def _advance_locked(self) -> None:
        if not self.running or self.ended:
            self.last_update = time.monotonic()
            return

        now = time.monotonic()
        elapsed = now - self.last_update
        ticks = int(elapsed)
        if ticks <= 0:
            return

        ticks = min(ticks, 60)
        for _ in range(ticks):
            self._advance_one_second_locked()
            if self.ended:
                break
        self.last_update += ticks

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            self._advance_locked()
            return {
                "running": bool(self.running),
                "ended": bool(self.ended),
                "rows_per_second": int(self.rows_per_second),
                "sim_seconds": int(self.sim_seconds),
                "cursor": int(self.cursor),
                "flows_processed": int(self.flows_processed),
                "alerts_detected": int(self.alerts_detected),
                "alerts_surfaced": int(self.alerts_surfaced),
                "first_alert_sim_second": self.first_alert_sim_second,
                "first_alert_row": self.first_alert_row,
                "recent_alerts": self.recent_alerts,
                "protocol_flow_counts": {k: int(v) for k, v in sorted(self.protocol_flow_counts.items())},
                "protocol_alert_counts": {k: int(v) for k, v in sorted(self.protocol_alert_counts.items())},
            }

    def session_summary(self) -> Dict[str, object]:
        with self.lock:
            flows_processed = int(self.flows_processed)
            alerts_surfaced = int(self.alerts_surfaced)
            alert_ratio = float(alerts_surfaced / flows_processed) if flows_processed > 0 else 0.0
            return {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_family": "CatBoost-E",
                "threshold_policy": str(self.threshold_policy),
                "inference_device_requested": str(self.inference_device_requested),
                "inference_device_active": str(self.prediction_task_type).lower(),
                "replay_mode": "simulator_replay",
                "replay_order": str(self.replay_order),
                "rows_per_second": int(self.rows_per_second),
                "running": bool(self.running),
                "ended": bool(self.ended),
                "sim_seconds": int(self.sim_seconds),
                "total_rows": int(self.total_rows),
                "flows_processed": flows_processed,
                "alerts_processed": alerts_surfaced,
                "alert_ratio": alert_ratio,
                "first_alert_sim_second": self.first_alert_sim_second,
                "first_alert_row": self.first_alert_row,
                "protocol_flow_counts": {k: int(v) for k, v in sorted(self.protocol_flow_counts.items())},
                "protocol_alert_counts": {k: int(v) for k, v in sorted(self.protocol_alert_counts.items())},
                "subset_actual_by_protocol": dict(self.subset_meta.get("actual_by_protocol", {})),
                "cache": dict(self.subset_meta.get("cache", {})),
            }

    def init_payload(self) -> Dict[str, object]:
        snap = self.snapshot()
        return {
            "meta": {
                "model_family": "CatBoost-E",
                "model_version": "E",
                "base_run_dir": str(self.base_run_dir),
                "matrix_run_dir": str(self.matrix_run_dir),
                "train_csv": str(self.train_csv),
                "test_csv": str(self.test_csv),
                "threshold_policy": str(self.threshold_policy),
                "inference_device_requested": str(self.inference_device_requested),
                "inference_device_active": str(self.prediction_task_type).lower(),
                "inference_device_note": str(self.inference_device_note),
                "replay_mode": "simulator_replay",
                "rows_per_second": int(self.rows_per_second),
                "flow_rate_min": 1,
                "flow_rate_max": 10,
                "demo_cache_mode": str(self.demo_cache_mode),
                "demo_cache_path": str(self.demo_cache_path),
                "demo_cache_source": str(self.subset_meta.get("cache", {}).get("source", "")),
                "replay_order": self.replay_order,
                "shuffle_seed": int(self.shuffle_seed),
                "local_top_n": int(self.local_top_n),
                "total_rows": int(self.total_rows),
                "thresholds_by_protocol": {
                    str(k): float(v) for k, v in sorted(self.thresholds_by_protocol.items())
                },
                "demo_subset": self.subset_meta,
            },
            "global_explanations": self.global_explanations,
            "state": snap,
        }


class IDSRequestHandler(BaseHTTPRequestHandler):
    runtime: LiveIDSRuntime | None = None

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _send_not_found(self) -> None:
        self._send_json({"error": "Not found"}, status=404)

    def _read_json_body(self) -> Dict[str, object]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            content_length = 0
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        runtime = IDSRequestHandler.runtime
        if runtime is None:
            self._send_json({"error": "Runtime not initialized"}, status=500)
            return

        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/init":
            self._send_json(runtime.init_payload())
            return
        if path == "/api/state":
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/session-summary":
            self._send_json({"summary": runtime.session_summary()})
            return
        self._send_not_found()

    def do_POST(self) -> None:
        runtime = IDSRequestHandler.runtime
        if runtime is None:
            self._send_json({"error": "Runtime not initialized"}, status=500)
            return

        path = urlparse(self.path).path
        if path == "/api/start":
            runtime.start()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/pause":
            runtime.pause()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/reset":
            runtime.reset()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/jump-first-alert":
            runtime.jump_to_first_alert()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/set-rate":
            payload = self._read_json_body()
            rate = safe_int(payload.get("rows_per_second"), 1)
            rate = runtime.set_rows_per_second(rate)
            self._send_json({"ok": True, "rows_per_second": int(rate), "state": runtime.snapshot()})
            return

        self._send_not_found()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--base-run-dir",
        type=str,
        default="reports/full_gpu_hpo_models_20260306_195851",
        help="Base run directory containing metrics_summary.json feature columns.",
    )
    parser.add_argument(
        "--matrix-run-dir",
        type=str,
        default=None,
        help="Robust matrix run containing CatBoost E model artifacts.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/merged/metadata_train.csv",
        help="Path to metadata_train.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/merged/metadata_test.csv",
        help="Path to metadata_test.csv",
    )
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="tuned_thresholds",
        choices=["tuned_thresholds", "baseline_thresholds"],
    )
    parser.add_argument(
        "--inference-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="CatBoost prediction device. auto tries GPU then falls back to CPU.",
    )
    parser.add_argument(
        "--rows-per-second",
        type=int,
        default=5,
        help="How many replay rows to score per simulated second (1-10).",
    )
    parser.add_argument(
        "--local-top-n",
        type=int,
        default=8,
        help="Top local explanation features for each surfaced alert.",
    )
    parser.add_argument(
        "--replay-order",
        type=str,
        default="interleave-protocol-source",
        choices=["sequential", "shuffle", "interleave-source", "interleave-protocol-source"],
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-recent-alerts", type=int, default=250)
    parser.add_argument("--demo-total-rows", type=int, default=9000)
    parser.add_argument("--demo-attack-ratio", type=float, default=0.7)
    parser.add_argument("--demo-seed", type=int, default=42)
    parser.add_argument(
        "--demo-cache-pickle",
        type=str,
        default=None,
        help="Optional prebuilt replay cache pickle. When set, startup skips rebuilding from train/test CSVs.",
    )
    parser.add_argument(
        "--demo-cache-mode",
        type=str,
        default="reuse",
        choices=["reuse", "refresh", "off"],
        help="reuse=load existing cache when available; refresh=rebuild and overwrite; off=no cache read/write.",
    )
    parser.add_argument(
        "--demo-cache-dir",
        type=str,
        default="reports/ids_replay_cache",
        help="Directory for cached in-memory demo subset artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_run_dir = Path(args.base_run_dir).resolve()
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Missing base run dir: {base_run_dir}")

    if args.matrix_run_dir:
        matrix_run_dir = Path(args.matrix_run_dir).resolve()
    else:
        latest = discover_latest_matrix_run("reports")
        if latest is None:
            raise RuntimeError("No robust matrix run found. Pass --matrix-run-dir explicitly.")
        matrix_run_dir = latest.resolve()

    if not matrix_run_dir.exists():
        raise FileNotFoundError(f"Missing matrix run dir: {matrix_run_dir}")

    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    demo_cache_pickle = Path(args.demo_cache_pickle).resolve() if args.demo_cache_pickle else None
    demo_cache_dir = Path(args.demo_cache_dir).resolve()

    runtime = LiveIDSRuntime(
        base_run_dir=base_run_dir,
        matrix_run_dir=matrix_run_dir,
        train_csv=train_csv,
        test_csv=test_csv,
        demo_cache_pickle=demo_cache_pickle,
        threshold_policy=args.threshold_policy,
        inference_device=args.inference_device,
        rows_per_second=args.rows_per_second,
        local_top_n=args.local_top_n,
        replay_order=args.replay_order,
        shuffle_seed=args.shuffle_seed,
        max_recent_alerts=args.max_recent_alerts,
        demo_total_rows=args.demo_total_rows,
        demo_attack_ratio=args.demo_attack_ratio,
        demo_seed=args.demo_seed,
        demo_cache_mode=args.demo_cache_mode,
        demo_cache_dir=demo_cache_dir,
    )
    IDSRequestHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), IDSRequestHandler)
    print(f"CatBoost-E IDS realtime API listening at http://{args.host}:{args.port}")
    print(
        "Rows loaded: "
        f"{runtime.total_rows} "
        f"| rows_per_second={runtime.rows_per_second} "
        f"| inference={runtime.prediction_task_type.lower()} "
        f"| threshold_policy={runtime.threshold_policy} "
        f"| cache_mode={runtime.demo_cache_mode} "
        f"| cache_hit={runtime.subset_meta.get('cache', {}).get('hit')} "
        f"| cache_source={runtime.subset_meta.get('cache', {}).get('source')}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
