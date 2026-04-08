                      

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def protocol_slug(protocol: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(protocol).strip().lower())
    slug = slug.strip("_")
    return slug if slug else "unknown"


def discover_latest_hpo_run(reports_dir: str | Path = "reports") -> Optional[Path]:
    base = Path(reports_dir)
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.glob("full_gpu_hpo_models_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_metrics_json(run_dir: str | Path) -> Dict:
    path = Path(run_dir) / "metrics_summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_columns(run_dir: str | Path) -> List[str]:
    metrics = load_metrics_json(run_dir)
    cols = metrics.get("feature_columns", [])
    if not isinstance(cols, list) or not cols:
        raise RuntimeError("Could not read feature_columns from metrics_summary.json")
    return [str(c) for c in cols]


def _parse_protocol_from_model_name(model_name: str) -> Optional[str]:
    if "__" not in model_name:
        return None
    stem = model_name.split("__", 1)[1]
    if stem.endswith(".json"):
        stem = stem[:-5]
    return protocol_slug(stem)


def load_xgb_models_by_protocol(run_dir: str | Path) -> Dict[str, xgb.Booster]:
    models_dir = Path(run_dir) / "models"
    if not models_dir.exists():
        raise RuntimeError(f"Missing models directory: {models_dir}")

    models: Dict[str, xgb.Booster] = {}
    for path in sorted(models_dir.glob("*__xgboost_tuned.json")):
        protocol = protocol_slug(path.name.split("__", 1)[0])
        booster = xgb.Booster()
        booster.load_model(str(path))
        models[protocol] = booster

    if not models:
        raise RuntimeError(f"No protocol XGBoost models found under {models_dir}")
    return models


def _extract_thresholds_from_metrics_json(metrics: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    per_protocol = metrics.get("results", {}).get("per_protocol_models", {})
    if not isinstance(per_protocol, dict):
        return out

    for proto, model_map in per_protocol.items():
        if not isinstance(model_map, dict):
            continue
        xgb_key = f"xgboost_tuned__{protocol_slug(proto)}"
        payload = model_map.get(xgb_key, {})
        protocol_test = payload.get("protocol_test", {}) if isinstance(payload, dict) else {}
        thr = protocol_test.get("threshold")
        try:
            thr_f = float(thr)
        except Exception:
            continue
        if math.isfinite(thr_f):
            out[protocol_slug(proto)] = thr_f
    return out


def load_thresholds_by_protocol(
    run_dir: str | Path,
    default_threshold: float = 0.5,
) -> Dict[str, float]:
    run_path = Path(run_dir)
    custom_path = run_path / "thresholds_by_protocol.json"
    csv_path = run_path / "metrics_summary_per_protocol_models.csv"
    metrics = load_metrics_json(run_path)
    from_json = _extract_thresholds_from_metrics_json(metrics)

    thresholds: Dict[str, float] = {}
    if custom_path.exists():
        try:
            with custom_path.open("r", encoding="utf-8") as f:
                custom_payload = json.load(f)
        except Exception:
            custom_payload = {}
        if isinstance(custom_payload, dict):
            for proto_raw, thr_raw in custom_payload.items():
                proto = protocol_slug(str(proto_raw))
                try:
                    thr_f = float(thr_raw)
                except Exception:
                    continue
                if math.isfinite(thr_f):
                    thresholds[proto] = thr_f

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "model" in df.columns:
            mask = df["model"].astype(str).str.startswith("xgboost_tuned__")
            xgb_rows = df.loc[mask].copy()
            for _, row in xgb_rows.iterrows():
                proto_raw = row.get("protocol_hint")
                model_name = str(row.get("model", ""))
                proto = protocol_slug(proto_raw) if pd.notna(proto_raw) else None
                if not proto:
                    parsed = _parse_protocol_from_model_name(model_name)
                    proto = parsed if parsed else "unknown"
                thr = pd.to_numeric(row.get("threshold"), errors="coerce")
                if pd.notna(thr) and np.isfinite(float(thr)):
                    thresholds.setdefault(proto, float(thr))

    for proto, thr in from_json.items():
        thresholds.setdefault(proto, thr)

    if not thresholds:
        model_protocols = list(load_xgb_models_by_protocol(run_path).keys())
        thresholds = {p: float(default_threshold) for p in model_protocols}
    else:
        for proto in list(thresholds.keys()):
            thr = thresholds[proto]
            if not np.isfinite(thr):
                thresholds[proto] = float(default_threshold)

    return thresholds


def choose_default_protocol(protocols: Iterable[str]) -> str:
    sorted_protocols = sorted([protocol_slug(p) for p in protocols])
    if "wifi" in sorted_protocols:
        return "wifi"
    return sorted_protocols[0] if sorted_protocols else "unknown"


def infer_protocol_labels(
    df: pd.DataFrame,
    known_protocols: Iterable[str],
    default_protocol: Optional[str] = None,
) -> np.ndarray:
    known = {protocol_slug(p) for p in known_protocols}
    fallback = protocol_slug(default_protocol) if default_protocol else choose_default_protocol(known)

    if "protocol_hint" in df.columns:
        raw = df["protocol_hint"].fillna("").astype(str).map(protocol_slug)
        values = raw.to_numpy()
    else:
        values = np.array([""] * len(df), dtype=object)

    routed = np.array([v if v in known else fallback for v in values], dtype=object)
    return routed


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    if df.empty:
        return np.empty((0, len(feature_columns)), dtype=np.float32)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    x_df = df.loc[:, feature_columns].copy()
    for col in feature_columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce").fillna(0.0)
    return x_df.to_numpy(dtype=np.float32, copy=False)


def routed_predict(
    df: pd.DataFrame,
    feature_columns: List[str],
    models_by_protocol: Dict[str, xgb.Booster],
    thresholds_by_protocol: Dict[str, float],
    default_protocol: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["protocol_used", "score_attack", "threshold", "prediction"])

    protocols = list(models_by_protocol.keys())
    fallback = protocol_slug(default_protocol) if default_protocol else choose_default_protocol(protocols)
    routed_protocol = infer_protocol_labels(df, protocols, default_protocol=fallback)
    x_mat = prepare_feature_matrix(df.copy(), feature_columns)

    score = np.zeros(len(df), dtype=np.float32)
    thr_out = np.zeros(len(df), dtype=np.float32)
    pred = np.zeros(len(df), dtype=np.int8)

    for proto in sorted(set(routed_protocol.tolist())):
        mask = routed_protocol == proto
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        model_proto = proto if proto in models_by_protocol else fallback
        model = models_by_protocol[model_proto]
        thr = float(thresholds_by_protocol.get(model_proto, 0.5))

        dmat = xgb.DMatrix(x_mat[idx])
        p = model.predict(dmat).astype(np.float32, copy=False)
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


def local_feature_contributions(
    booster: xgb.Booster,
    row_features: np.ndarray,
    feature_columns: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    row = np.asarray(row_features, dtype=np.float32).reshape(1, -1)
    contrib = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
    bias = float(contrib[-1])
    values = contrib[:-1]

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


def _feature_index_from_booster_key(key: str, feature_columns: List[str]) -> Optional[int]:
    if key in feature_columns:
        return feature_columns.index(key)
    if key.startswith("f") and key[1:].isdigit():
        idx = int(key[1:])
        if 0 <= idx < len(feature_columns):
            return idx
    return None


def global_feature_importance(booster: xgb.Booster, feature_columns: List[str]) -> pd.DataFrame:
    metrics = {
        "weight": booster.get_score(importance_type="weight"),
        "gain": booster.get_score(importance_type="gain"),
        "cover": booster.get_score(importance_type="cover"),
        "total_gain": booster.get_score(importance_type="total_gain"),
        "total_cover": booster.get_score(importance_type="total_cover"),
    }

    table = {"feature": feature_columns}
    for metric_name in metrics.keys():
        table[metric_name] = np.zeros(len(feature_columns), dtype=np.float64)

    for metric_name, raw in metrics.items():
        for key, value in raw.items():
            idx = _feature_index_from_booster_key(str(key), feature_columns)
            if idx is None:
                continue
            table[metric_name][idx] = float(value)

    out = pd.DataFrame(table)
    out = out.sort_values(["gain", "weight"], ascending=False).reset_index(drop=True)
    return out


def mean_abs_contrib_by_feature(
    booster: xgb.Booster,
    x_rows: np.ndarray,
    feature_columns: List[str],
    max_rows: int = 4096,
) -> pd.DataFrame:
    if x_rows.size == 0:
        return pd.DataFrame({"feature": feature_columns, "mean_abs_contribution": np.zeros(len(feature_columns))})

    rows = x_rows[: max(1, int(max_rows))]
    contrib = booster.predict(xgb.DMatrix(rows), pred_contribs=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_contribution": mean_abs.astype(np.float64, copy=False),
        }
    )
