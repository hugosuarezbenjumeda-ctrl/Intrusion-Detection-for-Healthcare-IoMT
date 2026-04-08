                      

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False


EXPECTED_METADATA_PREFIX = [
    "source_relpath",
    "source_filename",
    "source_modality",
    "source_group",
    "source_split_folder",
    "assigned_split",
    "split_strategy",
    "sample_name",
    "sample_core_name",
    "protocol_scope",
    "protocol_hint",
    "device",
    "scenario",
    "attack_name",
    "attack_family",
    "attack_variant",
    "is_attack",
    "is_benign",
    "label",
    "source_row_index",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "label",
    "is_attack",
    "is_benign",
    "attack_name",
    "attack_family",
    "attack_variant",
    "protocol_hint",
    "protocol_scope",
    "source_relpath",
    "source_filename",
    "source_row_index",
    "source_group",
    "source_split_folder",
    "assigned_split",
    "split_strategy",
    "sample_name",
    "sample_core_name",
    "device",
    "scenario",
}


def read_csv_header(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def load_feature_cols(train_csv: str) -> List[str]:
    header = read_csv_header(train_csv)
    if len(header) <= len(EXPECTED_METADATA_PREFIX):
        raise RuntimeError(
            f"Expected >{len(EXPECTED_METADATA_PREFIX)} columns in {train_csv}, got {len(header)}"
        )

    expected_prefix = EXPECTED_METADATA_PREFIX
    got_prefix = header[: len(expected_prefix)]
    if got_prefix != expected_prefix:
        raise RuntimeError(
            "CSV metadata prefix changed; refusing to rely on positional feature slicing. "
            f"Expected first columns {expected_prefix}, got {got_prefix}."
        )

    feature_cols = header[len(expected_prefix) :]
    leakage_cols = sorted(set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS)
    if leakage_cols:
        raise RuntimeError(
            "Potential leakage columns detected in feature set: "
            + ", ".join(leakage_cols)
        )
    if not feature_cols:
        raise RuntimeError("No feature columns detected after metadata prefix")
    if len(feature_cols) != len(set(feature_cols)):
        raise RuntimeError("Duplicate feature column names detected")
    return feature_cols


def assert_schema_has_columns(csv_path: str, required_cols: List[str]) -> None:
    header = read_csv_header(csv_path)
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")


def assert_train_test_capture_disjoint(
    train_source_relpath: pd.Series,
    test_source_relpath: pd.Series,
) -> Dict[str, Any]:
    train_paths: Set[str] = set(train_source_relpath.fillna("missing").astype(str).unique().tolist())
    test_paths: Set[str] = set(test_source_relpath.fillna("missing").astype(str).unique().tolist())
    overlap = sorted(train_paths & test_paths)
    if overlap:
        sample = overlap[:10]
        raise RuntimeError(
            "Train/test capture leakage detected: source_relpath overlap is non-zero. "
            f"overlap_count={len(overlap)} sample={sample}"
        )
    return {
        "train_unique_capture_paths": int(len(train_paths)),
        "test_unique_capture_paths": int(len(test_paths)),
        "overlap_capture_paths": 0,
    }


def build_protocol_class_val_mask(
    df: pd.DataFrame,
    seed: int,
    val_mod: int = 5,
    allow_row_fallback: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if val_mod <= 1:
        raise ValueError("val_mod must be >= 2")

    protocol = df["protocol_hint"].fillna("unknown").astype(str).to_numpy()
    y = df["label"].to_numpy(np.int8, copy=False)
    rel_hash = pd.util.hash_pandas_object(
        df["source_relpath"].astype("string"),
        index=False,
    ).to_numpy(np.uint64)
    row_idx = df["source_row_index"].to_numpy(np.uint64, copy=False)

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
                            "Validation split collapsed for single-file stratum and row fallback "
                            "is disabled. "
                            f"protocol={proto} class={cls} rows={n} files={n_files}"
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
                        "val_rows_after_split": int(stratum_val.sum()),
                        "train_rows_after_split": int(n - stratum_val.sum()),
                    }
                )

            val_mask[stratum_idx[stratum_val]] = True

    return val_mask, fallback_events


def stratified_subsample_indices(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    n = int(len(y))
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    attack_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]

                                                                       
    if len(benign_idx) >= max_rows:
        out = rng.choice(benign_idx, size=max_rows, replace=False).astype(np.int64, copy=False)
        rng.shuffle(out)
        return out

    max_attack = max_rows - len(benign_idx)
    take_attack = min(max_attack, len(attack_idx))
    attack_take = (
        rng.choice(attack_idx, size=take_attack, replace=False)
        if take_attack > 0
        else np.array([], dtype=np.int64)
    )
    out = np.concatenate([benign_idx, attack_take]).astype(np.int64, copy=False)
    rng.shuffle(out)
    return out


def select_threshold_f1_at_fpr(y_true: np.ndarray, proba: np.ndarray, max_fpr: float) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, proba)
    pos = float((y_true == 1).sum())
    neg = float((y_true == 0).sum())

    tp = tpr * pos
    fp = fpr * neg
    fn = pos - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)

    feasible_idx = np.where(fpr <= max_fpr)[0]
    if len(feasible_idx) > 0:
        best_idx = int(feasible_idx[np.argmax(f1[feasible_idx])])
        feasible = 1.0
    else:
        best_idx = int(np.argmin(fpr))
        feasible = 0.0

    return {
        "threshold": float(thr[best_idx]),
        "val_fpr": float(fpr[best_idx]),
        "val_recall": float(tpr[best_idx]),
        "val_precision": float(precision[best_idx]),
        "val_f1": float(f1[best_idx]),
        "feasible": float(feasible),
    }


def objective_from_val(y_val: np.ndarray, p_val: np.ndarray, max_fpr: float) -> Dict[str, float]:
    thr = select_threshold_f1_at_fpr(y_val, p_val, max_fpr=max_fpr)
    try:
        pr_auc = float(average_precision_score(y_val, p_val))
    except Exception:
        pr_auc = float("nan")
    fpr_gap = max(0.0, thr["val_fpr"] - max_fpr)

    max_fpr_safe = max(max_fpr, 1e-8)
    soft_fpr_penalty = 0.10 * (thr["val_fpr"] / max_fpr_safe)

                                                                                           
    objective = thr["val_f1"] + 0.05 * pr_auc - soft_fpr_penalty - 4.0 * fpr_gap
    if thr["feasible"] < 0.5:
        objective -= 0.10

    out = dict(thr)
    out.update(
        {
            "val_pr_auc": pr_auc,
            "objective": float(objective),
            "fpr_gap": float(fpr_gap),
            "soft_fpr_penalty": float(soft_fpr_penalty),
        }
    )
    return out


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, proba))
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, proba))
    except Exception:
        pr_auc = float("nan")
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def evaluate_with_thresholds(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    pred = (proba >= thresholds).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(y_true, proba))
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, proba))
    except Exception:
        pr_auc = float("nan")
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def eval_by_slice(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    group_name: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    g_series = pd.Series(groups).astype(str)
    for g in sorted(g_series.unique().tolist()):
        m = g_series.to_numpy() == g
        if int(m.sum()) == 0:
            continue
        y_slice = y_true[m]
        n_benign = int((y_slice == 0).sum())
        n_attack = int((y_slice == 1).sum())
        single_class = int(n_benign == 0 or n_attack == 0)
        out = evaluate(y_slice, proba[m], threshold)
        out.update(
            {
                "model": model_name,
                group_name: g,
                "n": int(m.sum()),
                "n_benign": n_benign,
                "n_attack": n_attack,
                "single_class_slice": single_class,
                "slice_warning": "single_class_slice" if single_class else "",
            }
        )
        rows.append(out)
    return rows


def eval_by_slice_with_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    groups: np.ndarray,
    group_name: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    g_series = pd.Series(groups).astype(str)
    for g in sorted(g_series.unique().tolist()):
        m = g_series.to_numpy() == g
        if int(m.sum()) == 0:
            continue
        y_slice = y_true[m]
        n_benign = int((y_slice == 0).sum())
        n_attack = int((y_slice == 1).sum())
        single_class = int(n_benign == 0 or n_attack == 0)
        out = evaluate_with_thresholds(y_slice, proba[m], thresholds[m])
        out.update(
            {
                "model": model_name,
                group_name: g,
                "n": int(m.sum()),
                "n_benign": n_benign,
                "n_attack": n_attack,
                "single_class_slice": single_class,
                "slice_warning": "single_class_slice" if single_class else "",
            }
        )
        rows.append(out)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_dynamic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        write_csv(path, [], ["empty"])
        return
    columns = sorted({k for r in rows for k in r.keys()})
    write_csv(path, rows, columns)


def protocol_slug(protocol: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in protocol.strip().lower())
    slug = slug.strip("_")
    return slug if slug else "unknown"


def protocol_seed_offset(protocol: str) -> int:
    return int(zlib.crc32(protocol.encode("utf-8")) & 0xFFFFFFFF)


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def clip_float(x: float, low: float, high: float) -> float:
    return float(min(max(x, low), high))


def clip_int(x: int, low: int, high: int) -> int:
    return int(min(max(x, low), high))


def weighted_choice(rng: np.random.Generator, values: List[Any]) -> Any:
    idx = int(rng.integers(0, len(values)))
    return values[idx]


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def sample_xgb_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.integers(900, 3201)),
        "max_depth": int(rng.integers(4, 13)),
        "learning_rate": sample_log_uniform(rng, 0.012, 0.16),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
        "min_child_weight": sample_log_uniform(rng, 0.8, 48.0),
        "reg_lambda": sample_log_uniform(rng, 1e-3, 80.0),
        "reg_alpha": sample_log_uniform(rng, 1e-4, 12.0),
        "gamma": sample_log_uniform(rng, 1e-4, 5.0),
        "max_bin": int(weighted_choice(rng, [256, 384, 512])),
    }


def mutate_xgb_params(base: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    p = dict(base)
    p["n_estimators"] = clip_int(int(round(base["n_estimators"] * rng.uniform(0.75, 1.30))), 700, 3600)
    p["max_depth"] = clip_int(base["max_depth"] + int(rng.integers(-2, 3)), 3, 14)
    p["learning_rate"] = clip_float(base["learning_rate"] * float(math.exp(rng.normal(0.0, 0.35))), 0.008, 0.22)
    p["subsample"] = clip_float(base["subsample"] + float(rng.normal(0.0, 0.08)), 0.55, 1.0)
    p["colsample_bytree"] = clip_float(base["colsample_bytree"] + float(rng.normal(0.0, 0.08)), 0.55, 1.0)
    p["min_child_weight"] = clip_float(base["min_child_weight"] * float(math.exp(rng.normal(0.0, 0.40))), 0.3, 64.0)
    p["reg_lambda"] = clip_float(base["reg_lambda"] * float(math.exp(rng.normal(0.0, 0.50))), 1e-4, 120.0)
    p["reg_alpha"] = clip_float(base["reg_alpha"] * float(math.exp(rng.normal(0.0, 0.60))), 1e-6, 24.0)
    p["gamma"] = clip_float(base["gamma"] * float(math.exp(rng.normal(0.0, 0.55))), 1e-6, 10.0)
    p["max_bin"] = int(weighted_choice(rng, [256, 384, 512]))
    return p


def run_xgb_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=n_jobs,
        early_stopping_rounds=120,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    p_val = model.predict_proba(x_val)[:, 1]
    best_it = int(getattr(model, "best_iteration", -1))
    return p_val, {"best_iteration": best_it}


def train_xgb_final(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Tuple[XGBClassifier, np.ndarray, np.ndarray]:
    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=n_jobs,
        early_stopping_rounds=160,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    return model, p_val, p_test


def sample_cat_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "iterations": int(rng.integers(1200, 4601)),
        "depth": int(rng.integers(5, 11)),
        "learning_rate": sample_log_uniform(rng, 0.012, 0.20),
        "l2_leaf_reg": sample_log_uniform(rng, 0.8, 36.0),
        "random_strength": sample_log_uniform(rng, 0.03, 8.0),
        "bagging_temperature": float(rng.uniform(0.0, 6.0)),
        "border_count": int(weighted_choice(rng, [64, 128, 254])),
    }


def mutate_cat_params(base: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    p = dict(base)
    p["iterations"] = clip_int(int(round(base["iterations"] * rng.uniform(0.70, 1.30))), 900, 5200)
    p["depth"] = clip_int(base["depth"] + int(rng.integers(-2, 3)), 4, 12)
    p["learning_rate"] = clip_float(base["learning_rate"] * float(math.exp(rng.normal(0.0, 0.35))), 0.008, 0.28)
    p["l2_leaf_reg"] = clip_float(base["l2_leaf_reg"] * float(math.exp(rng.normal(0.0, 0.45))), 0.4, 64.0)
    p["random_strength"] = clip_float(base["random_strength"] * float(math.exp(rng.normal(0.0, 0.55))), 0.01, 16.0)
    p["bagging_temperature"] = clip_float(base["bagging_temperature"] + float(rng.normal(0.0, 1.0)), 0.0, 8.0)
    p["border_count"] = int(weighted_choice(rng, [64, 128, 254]))
    return p


def run_cat_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=180,
        allow_writing_files=False,
        verbose=False,
        **params,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    p_val = model.predict_proba(x_val)[:, 1]
    best_it = int(model.get_best_iteration()) if model.get_best_iteration() is not None else -1
    return p_val, {"best_iteration": best_it}


def train_cat_final(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> Tuple[CatBoostClassifier, np.ndarray, np.ndarray]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=240,
        allow_writing_files=False,
        verbose=False,
        **params,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    return model, p_val, p_test


def sample_lgb_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.integers(1200, 5001)),
        "num_leaves": int(rng.integers(31, 321)),
        "max_depth": int(weighted_choice(rng, [-1, 6, 8, 10, 12, 14])),
        "learning_rate": sample_log_uniform(rng, 0.012, 0.18),
        "min_child_samples": int(rng.integers(20, 350)),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
        "reg_lambda": sample_log_uniform(rng, 1e-3, 80.0),
        "reg_alpha": sample_log_uniform(rng, 1e-4, 12.0),
        "min_split_gain": float(rng.uniform(0.0, 1.0)),
    }


def mutate_lgb_params(base: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    p = dict(base)
    p["n_estimators"] = clip_int(int(round(base["n_estimators"] * rng.uniform(0.70, 1.30))), 900, 5600)
    p["num_leaves"] = clip_int(base["num_leaves"] + int(rng.integers(-48, 49)), 16, 384)
    p["max_depth"] = int(weighted_choice(rng, [-1, 6, 8, 10, 12, 14]))
    p["learning_rate"] = clip_float(base["learning_rate"] * float(math.exp(rng.normal(0.0, 0.35))), 0.008, 0.24)
    p["min_child_samples"] = clip_int(base["min_child_samples"] + int(rng.integers(-60, 61)), 10, 500)
    p["subsample"] = clip_float(base["subsample"] + float(rng.normal(0.0, 0.08)), 0.55, 1.0)
    p["colsample_bytree"] = clip_float(base["colsample_bytree"] + float(rng.normal(0.0, 0.08)), 0.55, 1.0)
    p["reg_lambda"] = clip_float(base["reg_lambda"] * float(math.exp(rng.normal(0.0, 0.45))), 1e-4, 120.0)
    p["reg_alpha"] = clip_float(base["reg_alpha"] * float(math.exp(rng.normal(0.0, 0.50))), 1e-6, 24.0)
    p["min_split_gain"] = clip_float(base["min_split_gain"] + float(rng.normal(0.0, 0.2)), 0.0, 2.0)
    return p


def run_lgb_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not HAS_LIGHTGBM:
        raise RuntimeError("lightgbm import failed")
    callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        device="gpu",
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="average_precision",
        callbacks=callbacks,
    )
    p_val = model.predict_proba(x_val)[:, 1]
    best_it = int(model.best_iteration_) if model.best_iteration_ is not None else -1
    return p_val, {"best_iteration": best_it}


def train_lgb_final(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    if not HAS_LIGHTGBM:
        raise RuntimeError("lightgbm import failed")
    callbacks = [lgb.early_stopping(stopping_rounds=220, verbose=False)]
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        device="gpu",
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="average_precision",
        callbacks=callbacks,
    )
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    return model, p_val, p_test


def build_hidden_dims(width: int, depth: int) -> List[int]:
    dims: List[int] = []
    cur = width
    for _ in range(depth):
        dims.append(max(32, int(cur)))
        cur = max(32, int(round(cur * 0.60)))
    return dims


class TunedMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class NNTrainResult:
    val_proba: np.ndarray
    test_proba: Optional[np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    best_objective: float
    state_dict: Dict[str, torch.Tensor]


def sample_mlp_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "width": int(weighted_choice(rng, [256, 384, 512, 640])),
        "depth": int(weighted_choice(rng, [2, 3, 4])),
        "dropout": float(rng.uniform(0.08, 0.35)),
        "lr": sample_log_uniform(rng, 8e-5, 3e-3),
        "weight_decay": sample_log_uniform(rng, 1e-7, 2e-3),
        "batch_size": int(weighted_choice(rng, [16384, 32768, 65536])),
    }


def mutate_mlp_params(base: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    p = dict(base)
    p["width"] = int(weighted_choice(rng, [256, 384, 512, 640])) if rng.random() < 0.20 else base["width"]
    p["depth"] = int(weighted_choice(rng, [2, 3, 4])) if rng.random() < 0.25 else base["depth"]
    p["dropout"] = clip_float(base["dropout"] + float(rng.normal(0.0, 0.05)), 0.03, 0.45)
    p["lr"] = clip_float(base["lr"] * float(math.exp(rng.normal(0.0, 0.35))), 3e-5, 6e-3)
    p["weight_decay"] = clip_float(base["weight_decay"] * float(math.exp(rng.normal(0.0, 0.55))), 1e-8, 8e-3)
    p["batch_size"] = int(weighted_choice(rng, [16384, 32768, 65536]))
    return p


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: Optional[np.ndarray],
    params: Dict[str, Any],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    max_fpr: float,
) -> NNTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std if x_test is not None else None

    hidden_dims = build_hidden_dims(int(params["width"]), int(params["depth"]))
    model = TunedMLP(x_train.shape[1], hidden_dims=hidden_dims, dropout=float(params["dropout"])).to(device)

    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_n.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    x_val_t = torch.from_numpy(x_val_n.astype(np.float32)).to(device)
    x_test_t = torch.from_numpy(x_test_n.astype(np.float32)).to(device) if x_test_n is not None else None

    best_obj = -1e18
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_proba: Optional[np.ndarray] = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t).detach().cpu().numpy()
            val_proba = 1.0 / (1.0 + np.exp(-val_logits))

        stats = objective_from_val(y_val, val_proba, max_fpr=max_fpr)
        obj = float(stats["objective"])

        if obj > best_obj:
            best_obj = obj
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_proba = val_proba.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None or best_val_proba is None:
        raise RuntimeError("MLP training failed to produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.eval()

    test_proba: Optional[np.ndarray]
    if x_test_t is not None:
        with torch.no_grad():
            test_logits = model(x_test_t).detach().cpu().numpy()
        test_proba = 1.0 / (1.0 + np.exp(-test_logits))
    else:
        test_proba = None

    return NNTrainResult(
        val_proba=best_val_proba,
        test_proba=test_proba,
        mean=mean,
        std=std,
        best_epoch=best_epoch,
        best_objective=best_obj,
        state_dict=best_state,
    )


def iterative_tune(
    model_name: str,
    y_val: np.ndarray,
    max_fpr: float,
    coarse_trials: int,
    refine_trials: int,
    sampler,
    mutator,
    trial_runner,
    seed: int,
    clear_cuda: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], float]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_obj = -1e18
    trial_counter = 0

    def run_one(params: Dict[str, Any], stage: str) -> None:
        nonlocal best_params, best_obj, trial_counter
        trial_counter += 1
        start = time.perf_counter()
        row: Dict[str, Any] = {
            "model": model_name,
            "trial": int(trial_counter),
            "stage": stage,
            "params_json": json.dumps(params, sort_keys=True),
        }

        try:
            p_val, meta = trial_runner(params)
            stats = objective_from_val(y_val, p_val, max_fpr=max_fpr)
            elapsed = float(time.perf_counter() - start)

            row.update(
                {
                    "status": "ok",
                    "fit_seconds": elapsed,
                    "objective": float(stats["objective"]),
                    "val_f1": float(stats["val_f1"]),
                    "val_fpr": float(stats["val_fpr"]),
                    "val_precision": float(stats["val_precision"]),
                    "val_recall": float(stats["val_recall"]),
                    "val_pr_auc": float(stats["val_pr_auc"]),
                    "threshold": float(stats["threshold"]),
                    "fpr_gap": float(stats["fpr_gap"]),
                    "feasible": int(stats["feasible"]),
                }
            )
            for k, v in meta.items():
                row[f"meta_{k}"] = v

            if float(stats["objective"]) > best_obj:
                best_obj = float(stats["objective"])
                best_params = dict(params)

        except Exception as exc:                                 
            row.update(
                {
                    "status": "failed",
                    "fit_seconds": float(time.perf_counter() - start),
                    "error": str(exc)[:900],
                }
            )

        rows.append(row)
        gc.collect()
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    for _ in range(max(0, coarse_trials)):
        run_one(sampler(rng), stage="coarse")

    if best_params is not None:
        for _ in range(max(0, refine_trials)):
            run_one(mutator(best_params, rng), stage="refine")

    return best_params, rows, float(best_obj)


def tune_weighted_ensemble(
    val_prob_map: Dict[str, np.ndarray],
    y_val: np.ndarray,
    max_fpr: float,
    seed: int,
    n_iter: int,
) -> Optional[Dict[str, Any]]:
    model_names = sorted(val_prob_map.keys())
    if len(model_names) < 2:
        return None

    mat = np.vstack([val_prob_map[m] for m in model_names])
    rng = np.random.default_rng(seed)
    best: Optional[Dict[str, Any]] = None

    for _ in range(max(1, n_iter)):
        w = rng.dirichlet(np.ones(len(model_names), dtype=np.float64)).astype(np.float64)
        p_val = np.dot(w, mat)
        stats = objective_from_val(y_val, p_val, max_fpr=max_fpr)
        if best is None or float(stats["objective"]) > float(best["objective"]):
            best = {
                "models": model_names,
                "weights": [float(x) for x in w.tolist()],
                **stats,
            }

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_train.csv")
    ap.add_argument("--test-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_test.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default="/home/capstone15/reports")

    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--val-mod", type=int, default=5)
    ap.add_argument(
        "--disallow-row-fallback",
        action="store_true",
        help="Fail if validation splitting needs row-level fallback in single-file strata.",
    )
    ap.add_argument("--coarse-trials-boost", type=int, default=12)
    ap.add_argument("--refine-trials-boost", type=int, default=6)
    ap.add_argument("--coarse-trials-mlp", type=int, default=8)
    ap.add_argument("--refine-trials-mlp", type=int, default=4)
    ap.add_argument("--ensemble-iters", type=int, default=600)

    ap.add_argument("--tune-max-rows-boost", type=int, default=1200000)
    ap.add_argument("--tune-max-rows-mlp", type=int, default=700000)
    ap.add_argument("--mlp-tune-epochs", type=int, default=18)
    ap.add_argument("--mlp-final-epochs", type=int, default=28)
    ap.add_argument("--mlp-patience", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=16)

    args = ap.parse_args()
    np.random.seed(args.seed)

    feature_cols = load_feature_cols(args.train_csv)
    assert_schema_has_columns(
        args.test_csv,
        ["source_relpath", "label", "protocol_hint", "attack_family"] + feature_cols,
    )
    train_usecols = ["source_relpath", "source_row_index", "label", "protocol_hint"] + feature_cols
    test_usecols = ["source_relpath", "label", "protocol_hint", "attack_family"] + feature_cols

    dtypes = {c: "float32" for c in feature_cols}
    dtypes.update(
        {
            "source_relpath": "string",
            "label": "int8",
            "source_row_index": "int32",
            "protocol_hint": "string",
        }
    )

    train_df = pd.read_csv(args.train_csv, usecols=train_usecols, dtype=dtypes)
    val_mask, val_split_fallbacks = build_protocol_class_val_mask(
        train_df,
        args.seed,
        val_mod=args.val_mod,
        allow_row_fallback=not args.disallow_row_fallback,
    )
    x_all = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_all = train_df["label"].to_numpy(dtype=np.int8, copy=False)
    g_train_protocol = train_df["protocol_hint"].fillna("unknown").astype(str).to_numpy()

    test_df = pd.read_csv(
        args.test_csv,
        usecols=test_usecols,
        dtype={**dtypes, "attack_family": "string"},
    )
    overlap_summary = assert_train_test_capture_disjoint(
        train_df["source_relpath"],
        test_df["source_relpath"],
    )
    del train_df

    x_test_all = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test_all = test_df["label"].to_numpy(dtype=np.int8, copy=False)
    g_protocol = test_df["protocol_hint"].fillna("unknown").astype(str).to_numpy()
    g_family = test_df["attack_family"].fillna("unknown").astype(str).to_numpy()
    del test_df

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"full_gpu_hpo_models_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    total_train_rows = int((~val_mask).sum())
    total_val_rows = int(val_mask.sum())
    train_y_global = y_all[~val_mask]
    val_y_global = y_all[val_mask]

    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Dataset sizes: train={total_train_rows} val={total_val_rows} test={len(y_test_all)}")
    print(
        "[INFO] Capture-level disjointness: "
        f"train_unique={overlap_summary['train_unique_capture_paths']} "
        f"test_unique={overlap_summary['test_unique_capture_paths']} "
        f"overlap={overlap_summary['overlap_capture_paths']}"
    )
    if val_split_fallbacks:
        row_level_fb = sum(1 for e in val_split_fallbacks if e["fallback_type"] == "row_level_single_file")
        file_level_fb = sum(1 for e in val_split_fallbacks if e["fallback_type"] == "file_level_rebalance")
        print(
            "[WARN] Validation split fallback events detected: "
            f"file_level_rebalance={file_level_fb}, row_level_single_file={row_level_fb}"
        )
        for event in val_split_fallbacks[:8]:
            print(
                "[WARN] Split fallback "
                f"proto={event['protocol_hint']} label={event['label']} "
                f"rows={event['stratum_rows']} files={event['stratum_files']} "
                f"type={event['fallback_type']}"
            )

    protocol_values = sorted(set(g_train_protocol.tolist()) | set(g_protocol.tolist()))
    print(f"[INFO] Protocol slices: {protocol_values}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_tuning_rows: List[Dict[str, Any]] = []
    best_hparams_by_protocol: Dict[str, Dict[str, Dict[str, Any]]] = {}
    hpo_meta_by_protocol: Dict[str, Dict[str, Any]] = {}
    ensemble_meta_by_protocol: Dict[str, Any] = {}
    per_protocol_results: Dict[str, Any] = {}
    protocol_dataset_sizes: Dict[str, Any] = {}
    skipped_protocols: List[Dict[str, Any]] = []
    per_protocol_model_rows: List[Dict[str, Any]] = []

    routed_test_proba: Dict[str, np.ndarray] = {}
    routed_thresholds: Dict[str, np.ndarray] = {}
    routed_cover: Dict[str, np.ndarray] = {}
    routed_protocol_thresholds: Dict[str, Dict[str, Any]] = {}
    routed_protocol_val_stats: Dict[str, Dict[str, Any]] = {}

    protocol_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    single_class_test_slices: List[Dict[str, Any]] = []

    model_val_objective_num: Dict[str, float] = {}
    model_val_objective_den: Dict[str, int] = {}
    model_val_feasible_num: Dict[str, float] = {}

    for proto in protocol_values:
        proto_tag = protocol_slug(proto)
        seed_off = protocol_seed_offset(proto) % 100000

        mask_train_proto = g_train_protocol == proto
        mask_train_fit = mask_train_proto & (~val_mask)
        mask_val_fit = mask_train_proto & val_mask
        mask_test_proto = g_protocol == proto

        n_train_fit = int(mask_train_fit.sum())
        n_val_fit = int(mask_val_fit.sum())
        n_test = int(mask_test_proto.sum())

        if n_train_fit < 2 or n_val_fit < 2 or n_test < 1:
            skipped_protocols.append(
                {
                    "protocol_hint": proto,
                    "reason": "insufficient_rows",
                    "train_rows": n_train_fit,
                    "val_rows": n_val_fit,
                    "test_rows": n_test,
                }
            )
            continue

        x_train = x_all[mask_train_fit]
        y_train = y_all[mask_train_fit]
        x_val = x_all[mask_val_fit]
        y_val = y_all[mask_val_fit]
        x_test = x_test_all[mask_test_proto]
        y_test = y_test_all[mask_test_proto]
        g_protocol_test = g_protocol[mask_test_proto]
        g_family_test = g_family[mask_test_proto]
        n_test_benign = int((y_test == 0).sum())
        n_test_attack = int((y_test == 1).sum())
        if n_test_benign == 0 or n_test_attack == 0:
            single_class_test_slices.append(
                {
                    "scope": "protocol",
                    "model": "__all_models__",
                    "protocol_hint": proto,
                    "slice_name": proto,
                    "n": int(len(y_test)),
                    "n_benign": n_test_benign,
                    "n_attack": n_test_attack,
                }
            )

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            skipped_protocols.append(
                {
                    "protocol_hint": proto,
                    "reason": "single_class_train_or_val",
                    "train_rows": n_train_fit,
                    "val_rows": n_val_fit,
                    "test_rows": n_test,
                    "train_attack_ratio": float(y_train.mean()) if len(y_train) > 0 else float("nan"),
                    "val_attack_ratio": float(y_val.mean()) if len(y_val) > 0 else float("nan"),
                }
            )
            continue

        protocol_dataset_sizes[proto] = {
            "train": n_train_fit,
            "val": n_val_fit,
            "test": n_test,
            "train_attack_ratio": float(y_train.mean()),
            "val_attack_ratio": float(y_val.mean()),
            "test_attack_ratio": float(y_test.mean()),
        }

        idx_boost = stratified_subsample_indices(y_train, args.tune_max_rows_boost, args.seed + 101 + seed_off)
        idx_mlp = stratified_subsample_indices(y_train, args.tune_max_rows_mlp, args.seed + 151 + seed_off)

        x_train_boost = x_train[idx_boost]
        y_train_boost = y_train[idx_boost]
        x_train_mlp = x_train[idx_mlp]
        y_train_mlp = y_train[idx_mlp]

        print(
            "[INFO] "
            f"Protocol={proto}: train={n_train_fit}, val={n_val_fit}, test={n_test}, "
            f"tune_boost={len(y_train_boost)} ({y_train_boost.mean():.4f} attack), "
            f"tune_mlp={len(y_train_mlp)} ({y_train_mlp.mean():.4f} attack)"
        )

        tuning_rows: List[Dict[str, Any]] = []
        best_params: Dict[str, Dict[str, Any]] = {}
        tuning_meta: Dict[str, Any] = {}

        bx, rows_xgb, best_obj_xgb = iterative_tune(
            model_name="xgboost",
            y_val=y_val,
            max_fpr=args.target_fpr,
            coarse_trials=args.coarse_trials_boost,
            refine_trials=args.refine_trials_boost,
            sampler=sample_xgb_params,
            mutator=mutate_xgb_params,
            trial_runner=lambda p: run_xgb_trial(
                x_train_boost,
                y_train_boost,
                x_val,
                y_val,
                p,
                seed=args.seed + seed_off,
                n_jobs=args.n_jobs,
            ),
            seed=args.seed + 301 + seed_off,
        )
        for r in rows_xgb:
            r["protocol_hint"] = proto
        tuning_rows.extend(rows_xgb)
        if bx is not None:
            best_params["xgboost"] = bx
        tuning_meta["xgboost_best_objective"] = best_obj_xgb

        bc, rows_cat, best_obj_cat = iterative_tune(
            model_name="catboost",
            y_val=y_val,
            max_fpr=args.target_fpr,
            coarse_trials=args.coarse_trials_boost,
            refine_trials=args.refine_trials_boost,
            sampler=sample_cat_params,
            mutator=mutate_cat_params,
            trial_runner=lambda p: run_cat_trial(
                x_train_boost,
                y_train_boost,
                x_val,
                y_val,
                p,
                seed=args.seed + seed_off,
            ),
            seed=args.seed + 401 + seed_off,
        )
        for r in rows_cat:
            r["protocol_hint"] = proto
        tuning_rows.extend(rows_cat)
        if bc is not None:
            best_params["catboost"] = bc
        tuning_meta["catboost_best_objective"] = best_obj_cat

        if HAS_LIGHTGBM:
            bl, rows_lgb, best_obj_lgb = iterative_tune(
                model_name="lightgbm",
                y_val=y_val,
                max_fpr=args.target_fpr,
                coarse_trials=args.coarse_trials_boost,
                refine_trials=args.refine_trials_boost,
                sampler=sample_lgb_params,
                mutator=mutate_lgb_params,
                trial_runner=lambda p: run_lgb_trial(
                    x_train_boost,
                    y_train_boost,
                    x_val,
                    y_val,
                    p,
                    seed=args.seed + seed_off,
                    n_jobs=args.n_jobs,
                ),
                seed=args.seed + 501 + seed_off,
            )
            for r in rows_lgb:
                r["protocol_hint"] = proto
            tuning_rows.extend(rows_lgb)
            if bl is not None:
                best_params["lightgbm"] = bl
            tuning_meta["lightgbm_best_objective"] = best_obj_lgb
        else:
            tuning_rows.append(
                {
                    "model": "lightgbm",
                    "trial": 0,
                    "stage": "skipped",
                    "status": "failed",
                    "error": "lightgbm import unavailable",
                    "protocol_hint": proto,
                }
            )
            tuning_meta["lightgbm_best_objective"] = None

        bm, rows_mlp, best_obj_mlp = iterative_tune(
            model_name="complex_mlp",
            y_val=y_val,
            max_fpr=args.target_fpr,
            coarse_trials=args.coarse_trials_mlp,
            refine_trials=args.refine_trials_mlp,
            sampler=sample_mlp_params,
            mutator=mutate_mlp_params,
            trial_runner=lambda p: (
                train_mlp(
                    x_train=x_train_mlp,
                    y_train=y_train_mlp,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=None,
                    params=p,
                    seed=args.seed + seed_off,
                    device=device,
                    max_epochs=args.mlp_tune_epochs,
                    patience=args.mlp_patience,
                    max_fpr=args.target_fpr,
                ).val_proba,
                {},
            ),
            seed=args.seed + 601 + seed_off,
            clear_cuda=True,
        )
        for r in rows_mlp:
            r["protocol_hint"] = proto
        tuning_rows.extend(rows_mlp)
        if bm is not None:
            best_params["complex_mlp"] = bm
        tuning_meta["complex_mlp_best_objective"] = best_obj_mlp

        all_tuning_rows.extend(tuning_rows)
        best_hparams_by_protocol[proto] = best_params
        hpo_meta_by_protocol[proto] = tuning_meta

        all_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        model_meta: Dict[str, Any] = {}

        if "xgboost" in best_params:
            xgb_model, p_val_xgb, p_test_xgb = train_xgb_final(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                params=best_params["xgboost"],
                seed=args.seed + seed_off,
                n_jobs=args.n_jobs,
            )
            xgb_model.save_model(str(out_dir / "models" / f"{proto_tag}__xgboost_tuned.json"))
            all_scores["xgboost_tuned"] = (p_val_xgb, p_test_xgb)
            model_meta["xgboost_tuned"] = {"params": best_params["xgboost"]}
            del xgb_model
            gc.collect()

        if "catboost" in best_params:
            cat_model, p_val_cat, p_test_cat = train_cat_final(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                params=best_params["catboost"],
                seed=args.seed + seed_off,
            )
            cat_model.save_model(str(out_dir / "models" / f"{proto_tag}__catboost_tuned.cbm"))
            all_scores["catboost_tuned"] = (p_val_cat, p_test_cat)
            model_meta["catboost_tuned"] = {"params": best_params["catboost"]}
            del cat_model
            gc.collect()

        if "lightgbm" in best_params and HAS_LIGHTGBM:
            try:
                lgb_model, p_val_lgb, p_test_lgb = train_lgb_final(
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    x_test=x_test,
                    params=best_params["lightgbm"],
                    seed=args.seed + seed_off,
                    n_jobs=args.n_jobs,
                )
                lgb_model.booster_.save_model(str(out_dir / "models" / f"{proto_tag}__lightgbm_tuned.txt"))
                all_scores["lightgbm_tuned"] = (p_val_lgb, p_test_lgb)
                model_meta["lightgbm_tuned"] = {"params": best_params["lightgbm"]}
                del lgb_model
                gc.collect()
            except Exception as exc:
                model_meta["lightgbm_tuned"] = {
                    "params": best_params["lightgbm"],
                    "final_fit_error": str(exc),
                }

        if "complex_mlp" in best_params:
            mlp_final = train_mlp(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                params=best_params["complex_mlp"],
                seed=args.seed + seed_off,
                device=device,
                max_epochs=args.mlp_final_epochs,
                patience=args.mlp_patience,
                max_fpr=args.target_fpr,
            )
            if mlp_final.test_proba is None:
                raise RuntimeError("MLP final training did not produce test probabilities")

            torch.save(
                {
                    "state_dict": mlp_final.state_dict,
                    "mean": mlp_final.mean,
                    "std": mlp_final.std,
                    "best_epoch": mlp_final.best_epoch,
                    "best_objective": mlp_final.best_objective,
                    "params": best_params["complex_mlp"],
                },
                out_dir / "models" / f"{proto_tag}__complex_mlp_tuned_meta.pt",
            )
            all_scores["complex_mlp_tuned"] = (mlp_final.val_proba, mlp_final.test_proba)
            model_meta["complex_mlp_tuned"] = {
                "params": best_params["complex_mlp"],
                "best_epoch": mlp_final.best_epoch,
                "best_objective": mlp_final.best_objective,
            }

        if len(all_scores) < 1:
            skipped_protocols.append(
                {
                    "protocol_hint": proto,
                    "reason": "all_model_final_fits_failed",
                }
            )
            continue

        ensemble_meta = tune_weighted_ensemble(
            {k: v[0] for k, v in all_scores.items()},
            y_val=y_val,
            max_fpr=args.target_fpr,
            seed=args.seed + 901 + seed_off,
            n_iter=args.ensemble_iters,
        )
        if ensemble_meta is not None:
            names = ensemble_meta["models"]
            weights = np.array(ensemble_meta["weights"], dtype=np.float64)
            p_val_ens = np.zeros_like(next(iter(all_scores.values()))[0], dtype=np.float64)
            p_test_ens = np.zeros_like(next(iter(all_scores.values()))[1], dtype=np.float64)
            for i, n in enumerate(names):
                p_val_ens += weights[i] * all_scores[n][0]
                p_test_ens += weights[i] * all_scores[n][1]
            all_scores["weighted_ensemble"] = (p_val_ens.astype(np.float32), p_test_ens.astype(np.float32))
            ensemble_meta_by_protocol[proto] = ensemble_meta

        protocol_result_rows: Dict[str, Any] = {}
        for name, (p_val, p_test) in all_scores.items():
            val_stats = objective_from_val(y_val, p_val, max_fpr=args.target_fpr)
            test_metrics = evaluate(y_test, p_test, val_stats["threshold"])
            score = float(val_stats["objective"])

            model_label = f"{name}__{proto_tag}"
            row = dict(test_metrics)
            row.update(
                {
                    "model": model_label,
                    "protocol_hint": proto,
                    "selection_score": score,
                    "threshold": float(val_stats["threshold"]),
                    "val_objective": float(val_stats["objective"]),
                    "val_f1_at_selected_threshold": float(val_stats["val_f1"]),
                    "val_precision_at_selected_threshold": float(val_stats["val_precision"]),
                    "val_recall_at_selected_threshold": float(val_stats["val_recall"]),
                    "val_fpr_at_selected_threshold": float(val_stats["val_fpr"]),
                    "val_threshold_feasible": int(val_stats["feasible"]),
                }
            )
            per_protocol_model_rows.append(row)

            p_rows = eval_by_slice(y_test, p_test, val_stats["threshold"], g_protocol_test, "protocol_hint", model_label)
            f_rows = eval_by_slice(y_test, p_test, val_stats["threshold"], g_family_test, "attack_family", model_label)
            protocol_rows.extend(p_rows)
            family_rows.extend(f_rows)
            for r in p_rows:
                if int(r.get("single_class_slice", 0)) == 1:
                    single_class_test_slices.append(
                        {
                            "scope": "protocol_slice_metric",
                            "model": model_label,
                            "protocol_hint": proto,
                            "slice_name": str(r.get("protocol_hint")),
                            "n": int(r.get("n", 0)),
                            "n_benign": int(r.get("n_benign", 0)),
                            "n_attack": int(r.get("n_attack", 0)),
                        }
                    )
            for r in f_rows:
                if int(r.get("single_class_slice", 0)) == 1:
                    single_class_test_slices.append(
                        {
                            "scope": "attack_family_slice_metric",
                            "model": model_label,
                            "protocol_hint": proto,
                            "slice_name": str(r.get("attack_family")),
                            "n": int(r.get("n", 0)),
                            "n_benign": int(r.get("n_benign", 0)),
                            "n_attack": int(r.get("n_attack", 0)),
                        }
                    )

            protocol_result_rows[model_label] = {
                "protocol_test": row,
                "protocol_slices": p_rows,
                "attack_family_slices": f_rows,
                "model_meta": model_meta.get(name, {}),
            }

            model_val_objective_num[name] = model_val_objective_num.get(name, 0.0) + float(val_stats["objective"]) * float(n_val_fit)
            model_val_objective_den[name] = model_val_objective_den.get(name, 0) + int(n_val_fit)
            model_val_feasible_num[name] = model_val_feasible_num.get(name, 0.0) + float(val_stats["feasible"]) * float(n_val_fit)

            if name not in routed_test_proba:
                routed_test_proba[name] = np.zeros(len(y_test_all), dtype=np.float32)
                routed_thresholds[name] = np.zeros(len(y_test_all), dtype=np.float32)
                routed_cover[name] = np.zeros(len(y_test_all), dtype=bool)
                routed_protocol_thresholds[name] = {}
                routed_protocol_val_stats[name] = {}

            routed_test_proba[name][mask_test_proto] = p_test.astype(np.float32, copy=False)
            routed_thresholds[name][mask_test_proto] = float(val_stats["threshold"])
            routed_cover[name][mask_test_proto] = True
            routed_protocol_thresholds[name][proto] = {
                "threshold": float(val_stats["threshold"]),
                "val_f1": float(val_stats["val_f1"]),
                "val_fpr": float(val_stats["val_fpr"]),
                "val_threshold_feasible": int(val_stats["feasible"]),
            }
            routed_protocol_val_stats[name][proto] = {
                "val_objective": float(val_stats["objective"]),
                "val_pr_auc": float(val_stats["val_pr_auc"]),
            }

        per_protocol_results[proto] = protocol_result_rows

    if single_class_test_slices:
        dedup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for item in single_class_test_slices:
            key = (
                str(item.get("scope", "")),
                str(item.get("model", "")),
                str(item.get("protocol_hint", "")),
                str(item.get("slice_name", "")),
            )
            dedup[key] = item
        single_class_test_slices = list(dedup.values())
        print(f"[WARN] Single-class test slice flags: {len(single_class_test_slices)}")

    write_dynamic_csv(out_dir / "hpo_trials.csv", all_tuning_rows)
    with (out_dir / "best_hparams.json").open("w", encoding="utf-8") as f:
        json.dump(best_hparams_by_protocol, f, indent=2)
    with (out_dir / "ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump(ensemble_meta_by_protocol, f, indent=2)

    summary_rows: List[Dict[str, Any]] = []
    global_routed_results: Dict[str, Any] = {}
    for name in sorted(routed_test_proba.keys()):
        cover = routed_cover[name]
        covered_n = int(cover.sum())
        if covered_n < 1:
            continue

        p_test = routed_test_proba[name][cover]
        t_test = routed_thresholds[name][cover]
        y_test = y_test_all[cover]
        g_protocol_cov = g_protocol[cover]
        g_family_cov = g_family[cover]

        test_metrics = evaluate_with_thresholds(y_test, p_test, t_test)
        val_rows_for_selection = int(model_val_objective_den.get(name, 0))
        if val_rows_for_selection < 1:
            continue
        score = float(model_val_objective_num[name] / float(val_rows_for_selection))
        val_feasible_ratio = float(model_val_feasible_num.get(name, 0.0) / float(val_rows_for_selection))
        model_label = f"{name}__protocol_routed"
        row = dict(test_metrics)
        row.update(
            {
                "model": model_label,
                "selection_score": score,
                "val_weighted_objective": score,
                "val_feasible_ratio_for_selection": val_feasible_ratio,
                "val_rows_for_selection": val_rows_for_selection,
                "threshold": float("nan"),
                "val_f1_at_selected_threshold": float("nan"),
                "val_precision_at_selected_threshold": float("nan"),
                "val_recall_at_selected_threshold": float("nan"),
                "val_fpr_at_selected_threshold": float("nan"),
                "val_threshold_feasible": int(1 if covered_n == len(y_test_all) else 0),
            }
        )
        summary_rows.append(row)

        p_rows = eval_by_slice_with_thresholds(
            y_test,
            p_test,
            t_test,
            g_protocol_cov,
            "protocol_hint",
            model_label,
        )
        f_rows = eval_by_slice_with_thresholds(
            y_test,
            p_test,
            t_test,
            g_family_cov,
            "attack_family",
            model_label,
        )
        protocol_rows.extend(p_rows)
        family_rows.extend(f_rows)
        for r in p_rows:
            if int(r.get("single_class_slice", 0)) == 1:
                single_class_test_slices.append(
                    {
                        "scope": "protocol_slice_metric",
                        "model": model_label,
                        "protocol_hint": str(r.get("protocol_hint")),
                        "slice_name": str(r.get("protocol_hint")),
                        "n": int(r.get("n", 0)),
                        "n_benign": int(r.get("n_benign", 0)),
                        "n_attack": int(r.get("n_attack", 0)),
                    }
                )
        for r in f_rows:
            if int(r.get("single_class_slice", 0)) == 1:
                single_class_test_slices.append(
                    {
                        "scope": "attack_family_slice_metric",
                        "model": model_label,
                        "protocol_hint": "",
                        "slice_name": str(r.get("attack_family")),
                        "n": int(r.get("n", 0)),
                        "n_benign": int(r.get("n_benign", 0)),
                        "n_attack": int(r.get("n_attack", 0)),
                    }
                )

        global_routed_results[model_label] = {
            "coverage_rows": covered_n,
            "coverage_ratio": float(covered_n / max(1, len(y_test_all))),
            "global_test": row,
            "protocol_slices": p_rows,
            "attack_family_slices": f_rows,
            "per_protocol_thresholds": routed_protocol_thresholds.get(name, {}),
            "per_protocol_val_stats": routed_protocol_val_stats.get(name, {}),
        }

    if len(summary_rows) < 1:
        raise RuntimeError("No protocol-routed model predictions were produced")

    summary_rows.sort(
        key=lambda r: float(r["selection_score"])
        if np.isfinite(float(r["selection_score"]))
        else float("-inf"),
        reverse=True,
    )
    best_model = summary_rows[0]["model"]

    write_csv(
        out_dir / "metrics_summary.csv",
        summary_rows,
        [
            "model",
            "selection_score",
            "val_weighted_objective",
            "val_feasible_ratio_for_selection",
            "val_rows_for_selection",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
            "val_f1_at_selected_threshold",
            "val_precision_at_selected_threshold",
            "val_recall_at_selected_threshold",
            "val_fpr_at_selected_threshold",
            "val_threshold_feasible",
        ],
    )
    write_csv(
        out_dir / "metrics_summary_per_protocol_models.csv",
        per_protocol_model_rows,
        [
            "protocol_hint",
            "model",
            "selection_score",
            "val_objective",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
            "val_f1_at_selected_threshold",
            "val_precision_at_selected_threshold",
            "val_recall_at_selected_threshold",
            "val_fpr_at_selected_threshold",
            "val_threshold_feasible",
        ],
    )
    write_csv(
        out_dir / "slice_metrics_protocol.csv",
        protocol_rows,
        [
            "model",
            "protocol_hint",
            "n",
            "n_benign",
            "n_attack",
            "single_class_slice",
            "slice_warning",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    write_csv(
        out_dir / "slice_metrics_attack_family.csv",
        family_rows,
        [
            "model",
            "attack_family",
            "n",
            "n_benign",
            "n_attack",
            "single_class_slice",
            "slice_warning",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "seed": args.seed,
                "target_fpr": args.target_fpr,
                "validation_split": {
                    "val_mod": args.val_mod,
                    "disallow_row_fallback": bool(args.disallow_row_fallback),
                    "fallback_events": val_split_fallbacks,
                },
                "feature_columns": feature_cols,
                "protocol_slices": protocol_values,
                "dataset_sizes": {
                    "train": total_train_rows,
                    "val": total_val_rows,
                    "test": int(len(y_test_all)),
                    "train_attack_ratio": float(train_y_global.mean()),
                    "val_attack_ratio": float(val_y_global.mean()),
                    "test_attack_ratio": float(y_test_all.mean()),
                    "per_protocol": protocol_dataset_sizes,
                },
                "data_integrity_checks": {
                    "capture_disjointness": overlap_summary,
                },
                "hpo": {
                    "coarse_trials_boost": args.coarse_trials_boost,
                    "refine_trials_boost": args.refine_trials_boost,
                    "coarse_trials_mlp": args.coarse_trials_mlp,
                    "refine_trials_mlp": args.refine_trials_mlp,
                    "ensemble_iters": args.ensemble_iters,
                    "best_params_by_protocol": best_hparams_by_protocol,
                    "best_objectives_by_protocol": hpo_meta_by_protocol,
                },
                "skipped_protocols": skipped_protocols,
                "single_class_test_slices": single_class_test_slices,
                "best_model_by_selection_score": best_model,
                "results": {
                    "global_protocol_routed": global_routed_results,
                    "per_protocol_models": per_protocol_results,
                },
            },
            f,
            indent=2,
        )

    with (out_dir / "RUN_SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Iterative GPU HPO training completed (individual model per protocol).\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write("Model selection policy: validation-only objective (no test-based ranking).\n")
        f.write(f"Best model by validation selection score: {best_model}\n")
        f.write(f"Rows train={total_train_rows} val={total_val_rows} test={len(y_test_all)}\n")
        f.write(f"Protocols={','.join(protocol_values)}\n")
        f.write(f"Features={len(feature_cols)}\n")
        f.write(
            "Capture disjointness: "
            f"train_unique={overlap_summary['train_unique_capture_paths']} "
            f"test_unique={overlap_summary['test_unique_capture_paths']} "
            f"overlap={overlap_summary['overlap_capture_paths']}\n"
        )
        f.write(f"Validation fallback events={len(val_split_fallbacks)}\n")
        f.write(f"Single-class test slice flags={len(single_class_test_slices)}\n")

    print(str(out_dir))


if __name__ == "__main__":
    main()
