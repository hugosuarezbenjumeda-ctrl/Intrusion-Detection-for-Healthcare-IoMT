
                      

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    HAS_TORCH = False

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

ALLOWED_MATRIX_MODELS = {"xgboost", "catboost"}


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


def parse_int_csv(raw: str, *, allow_empty: bool = False) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out and not allow_empty:
        raise ValueError("Empty int list")
    return out


def parse_str_csv(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(t)
    if not out:
        raise ValueError("Empty string list")
    return out


def _optional_positive_int(raw: Optional[int]) -> Optional[int]:
    if raw is None:
        return None
    val = int(raw)
    if val <= 0:
        return None
    return val


def _resolve_int_override(stage_raw: Optional[int], shared_default: int) -> int:
    if stage_raw is None:
        return int(shared_default)
    return int(stage_raw)


def _resolve_float_override(stage_raw: Optional[float], shared_default: float) -> float:
    if stage_raw is None:
        return float(shared_default)
    return float(stage_raw)


def _optional_nonnegative_int(raw: Optional[int]) -> Optional[int]:
    if raw is None:
        return None
    val = int(raw)
    if val <= 0:
        return None
    return val


def _resolve_optional_int_override(stage_raw: Optional[int], shared_default: Optional[int]) -> Optional[int]:
    if stage_raw is None:
        return _optional_nonnegative_int(shared_default)
    return _optional_nonnegative_int(stage_raw)


def _resolve_adaptive_sample_target(
    *,
    pool_size: int,
    fraction: float,
    min_rows: Optional[int],
    max_rows: Optional[int],
) -> int:
    n = int(max(0, int(pool_size)))
    if n <= 0:
        return 0
    frac = max(0.0, float(fraction))
    target = int(round(float(n) * frac))
    if min_rows is not None and int(min_rows) > 0:
        target = max(target, int(min_rows))
    if max_rows is not None and int(max_rows) > 0:
        target = min(target, int(max_rows))
    target = max(1, int(target))
    return int(min(n, target))


def _resolve_stage_search_profile(args: argparse.Namespace, stage_name: str) -> Dict[str, Any]:
    stage = str(stage_name).strip().lower()
    if stage not in {"coarse", "stability"}:
        raise ValueError(f"Unsupported stage_name={stage_name}")
    prefix = "coarse" if stage == "coarse" else "stability"
    sampling_policy = str(getattr(args, "sampling_policy", "fixed_count")).strip().lower()
    if sampling_policy not in {"fixed_count", "adaptive_fraction"}:
        raise ValueError(f"Unsupported sampling_policy={sampling_policy}")

    shared_query_budget_benign = int(args.hardneg_query_budget)
    shared_query_max_steps_benign = int(args.hardneg_query_max_steps)
    shared_query_candidates_benign = int(args.hardneg_candidates_per_step)
    shared_query_score_batch_benign = int(args.query_score_batch_rows)
    shared_query_budget_mal = (
        int(args.val_malicious_query_budget)
        if args.val_malicious_query_budget is not None
        else shared_query_budget_benign
    )
    shared_query_max_steps_mal = (
        int(args.val_malicious_query_max_steps)
        if args.val_malicious_query_max_steps is not None
        else shared_query_max_steps_benign
    )
    shared_query_candidates_mal = (
        int(args.val_malicious_candidates_per_step)
        if args.val_malicious_candidates_per_step is not None
        else shared_query_candidates_benign
    )
    shared_query_score_batch_mal = (
        int(args.val_malicious_score_batch_rows)
        if args.val_malicious_score_batch_rows is not None
        else shared_query_score_batch_benign
    )

    shared_max_active_benign = _optional_positive_int(args.hardneg_query_max_active_rows_per_step)
    shared_stagnation_patience_benign = _optional_positive_int(args.hardneg_query_stagnation_patience)
    shared_stagnation_min_delta_benign = float(args.hardneg_query_stagnation_min_delta)
    shared_max_active_mal = (
        _optional_positive_int(args.val_malicious_query_max_active_rows_per_step)
        if args.val_malicious_query_max_active_rows_per_step is not None
        else shared_max_active_benign
    )
    shared_stagnation_patience_mal = (
        _optional_positive_int(args.val_malicious_query_stagnation_patience)
        if args.val_malicious_query_stagnation_patience is not None
        else shared_stagnation_patience_benign
    )
    shared_stagnation_min_delta_mal = (
        float(args.val_malicious_query_stagnation_min_delta)
        if args.val_malicious_query_stagnation_min_delta is not None
        else shared_stagnation_min_delta_benign
    )

    bucket_specs = {
        "train_benign": {
            "fixed_key": "hardneg_train_benign_sample",
            "fraction_key": "hardneg_train_benign_fraction",
            "min_key": "hardneg_train_benign_min",
            "max_key": "hardneg_train_benign_max",
        },
        "train_malicious": {
            "fixed_key": "hardneg_train_malicious_sample",
            "fraction_key": "hardneg_train_malicious_fraction",
            "min_key": "hardneg_train_malicious_min",
            "max_key": "hardneg_train_malicious_max",
        },
        "val_benign": {
            "fixed_key": "hardneg_val_benign_sample",
            "fraction_key": "hardneg_val_benign_fraction",
            "min_key": "hardneg_val_benign_min",
            "max_key": "hardneg_val_benign_max",
        },
        "val_malicious": {
            "fixed_key": "val_malicious_sample",
            "fraction_key": "val_malicious_fraction",
            "min_key": "val_malicious_min",
            "max_key": "val_malicious_max",
        },
    }

    sampling_buckets: Dict[str, Dict[str, Any]] = {}
    for bucket_name, spec in bucket_specs.items():
        fixed_key = str(spec["fixed_key"])
        fraction_key = str(spec["fraction_key"])
        min_key = str(spec["min_key"])
        max_key = str(spec["max_key"])

        fixed_count = _resolve_int_override(
            getattr(args, f"{prefix}_{fixed_key}"),
            int(getattr(args, fixed_key)),
        )
        fraction = _resolve_float_override(
            getattr(args, f"{prefix}_{fraction_key}"),
            float(getattr(args, fraction_key)),
        )
        min_rows = _resolve_optional_int_override(
            getattr(args, f"{prefix}_{min_key}"),
            _optional_nonnegative_int(getattr(args, min_key)),
        )
        max_rows = _resolve_optional_int_override(
            getattr(args, f"{prefix}_{max_key}"),
            _optional_nonnegative_int(getattr(args, max_key)),
        )
        if min_rows is not None and max_rows is not None and int(max_rows) < int(min_rows):
            max_rows = int(min_rows)

        sampling_buckets[bucket_name] = {
            "fixed_count": int(fixed_count),
            "fraction": float(fraction),
            "min_rows": min_rows,
            "max_rows": max_rows,
        }

    profile: Dict[str, Any] = {
        "stage": stage,
        "sampling_policy": sampling_policy,
        "sampling_buckets": sampling_buckets,
        "hardneg_train_benign_sample": int(sampling_buckets["train_benign"]["fixed_count"]),
        "hardneg_train_malicious_sample": int(sampling_buckets["train_malicious"]["fixed_count"]),
        "hardneg_val_benign_sample": int(sampling_buckets["val_benign"]["fixed_count"]),
        "val_malicious_sample": int(sampling_buckets["val_malicious"]["fixed_count"]),
        "hardneg_train_benign_fraction": float(sampling_buckets["train_benign"]["fraction"]),
        "hardneg_train_benign_min": sampling_buckets["train_benign"]["min_rows"],
        "hardneg_train_benign_max": sampling_buckets["train_benign"]["max_rows"],
        "hardneg_train_malicious_fraction": float(sampling_buckets["train_malicious"]["fraction"]),
        "hardneg_train_malicious_min": sampling_buckets["train_malicious"]["min_rows"],
        "hardneg_train_malicious_max": sampling_buckets["train_malicious"]["max_rows"],
        "hardneg_val_benign_fraction": float(sampling_buckets["val_benign"]["fraction"]),
        "hardneg_val_benign_min": sampling_buckets["val_benign"]["min_rows"],
        "hardneg_val_benign_max": sampling_buckets["val_benign"]["max_rows"],
        "val_malicious_fraction": float(sampling_buckets["val_malicious"]["fraction"]),
        "val_malicious_min": sampling_buckets["val_malicious"]["min_rows"],
        "val_malicious_max": sampling_buckets["val_malicious"]["max_rows"],
        "query_budget_benign": _resolve_int_override(
            getattr(args, f"{prefix}_hardneg_query_budget"),
            shared_query_budget_benign,
        ),
        "query_max_steps_benign": _resolve_int_override(
            getattr(args, f"{prefix}_hardneg_query_max_steps"),
            shared_query_max_steps_benign,
        ),
        "query_candidates_benign": _resolve_int_override(
            getattr(args, f"{prefix}_hardneg_candidates_per_step"),
            shared_query_candidates_benign,
        ),
        "query_subset_benign": int(args.hardneg_feature_subset_size),
        "query_score_batch_benign": _resolve_int_override(
            getattr(args, f"{prefix}_query_score_batch_rows"),
            shared_query_score_batch_benign,
        ),
        "query_fast_projection": bool(args.query_fast_projection),
        "query_refine_topk": max(
            1,
            _resolve_int_override(
                getattr(args, f"{prefix}_query_refine_topk"),
                int(args.query_refine_topk),
            ),
        ),
        "query_budget_mal": _resolve_int_override(
            getattr(args, f"{prefix}_val_malicious_query_budget"),
            shared_query_budget_mal,
        ),
        "query_max_steps_mal": _resolve_int_override(
            getattr(args, f"{prefix}_val_malicious_query_max_steps"),
            shared_query_max_steps_mal,
        ),
        "query_candidates_mal": _resolve_int_override(
            getattr(args, f"{prefix}_val_malicious_candidates_per_step"),
            shared_query_candidates_mal,
        ),
        "query_subset_mal": (
            int(args.val_malicious_feature_subset_size)
            if args.val_malicious_feature_subset_size is not None
            else int(args.hardneg_feature_subset_size)
        ),
        "query_score_batch_mal": _resolve_int_override(
            getattr(args, f"{prefix}_val_malicious_score_batch_rows"),
            shared_query_score_batch_mal,
        ),
        "query_max_active_rows_benign": shared_max_active_benign,
        "query_stagnation_patience_benign": shared_stagnation_patience_benign,
        "query_stagnation_min_delta_benign": float(shared_stagnation_min_delta_benign),
        "query_max_active_rows_mal": shared_max_active_mal,
        "query_stagnation_patience_mal": shared_stagnation_patience_mal,
        "query_stagnation_min_delta_mal": float(shared_stagnation_min_delta_mal),
    }
    return profile


def _resolve_stage_models(
    args: argparse.Namespace,
    *,
    stage_name: str,
    shortlist_models: Optional[Sequence[str]] = None,
) -> List[str]:
    stage = str(stage_name).strip().lower()
    if stage == "coarse":
        raw = str(getattr(args, "coarse_models", "")).strip()
        if not raw:
            out = [str(m).strip().lower() for m in parse_str_csv(args.models)]
        else:
            out = [str(m).strip().lower() for m in parse_str_csv(raw)]
        return _validate_matrix_model_list(out, context="coarse stage models")
    if stage == "stability":
        raw = str(getattr(args, "stability_models", "")).strip()
        if raw:
            out = [str(m).strip().lower() for m in parse_str_csv(raw)]
            return _validate_matrix_model_list(out, context="stability stage models")
        if shortlist_models is None:
            return []
        seen = set()
        out: List[str] = []
        for m in shortlist_models:
            name = str(m).strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return _validate_matrix_model_list(out, context="stability shortlist models")
    raise ValueError(f"Unsupported stage_name={stage_name}")


def _validate_matrix_model_list(model_names: Sequence[str], *, context: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in model_names:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not out:
        raise RuntimeError(f"{context}: empty model list.")
    unsupported = [m for m in out if m not in ALLOWED_MATRIX_MODELS]
    if unsupported:
        raise RuntimeError(
            f"{context}: unsupported model(s) {unsupported}. "
            f"Allowed models={sorted(ALLOWED_MATRIX_MODELS)}"
        )
    return out


def _stable_protocol_offset(proto: str) -> int:
    s = str(proto)
                                                                                        
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 10_000)


def _runtime_thread_count(default_threads: int = 8) -> int:
    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        raw = os.getenv(key, "").strip()
        if raw:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except Exception:
                pass
    try:
        cpu_count = int(os.cpu_count() or default_threads)
    except Exception:
        cpu_count = int(default_threads)
    return max(1, cpu_count)


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
    ap.add_argument("--protocols", type=str, default="wifi,mqtt,bluetooth")
    ap.add_argument("--models", type=str, default="xgboost,catboost")
    ap.add_argument(
        "--coarse-models",
        type=str,
        default="xgboost,catboost",
        help="Model list for coarse stage. Empty means fall back to --models.",
    )
    ap.add_argument(
        "--stability-models",
        type=str,
        default="",
        help="Optional model list for stability stage. Empty means use shortlisted coarse models only.",
    )
    ap.add_argument(
        "--attack-source-mode",
        choices=("fixed_xgb", "candidate_model", "hybrid"),
        default="fixed_xgb",
        help="Attack scorer source. 'hybrid' uses fixed_xgb for coarse and candidate_model for stability.",
    )
    ap.add_argument("--stage-mode", choices=("coarse", "stability", "both"), default="both")
    ap.add_argument("--coarse-run-dir", type=str, default=None)
    ap.add_argument("--coarse-seed", type=int, default=42)
    ap.add_argument("--stability-seeds", type=str, default="43,44,45,46,47,48,49,50")
    ap.add_argument(
        "--stability-split-mode",
        choices=("shared_val_seed", "per_seed"),
        default="shared_val_seed",
        help="Validation split policy for stability stage: shared split seed across all stability seeds, or per-seed split.",
    )
    ap.add_argument(
        "--stability-sampling-mode",
        choices=("fixed", "per_seed", "hybrid"),
        default="fixed",
        help="Stability-stage sampling seed policy: fixed across seeds, per-seed, or hybrid (fixed for bluetooth only).",
    )
    ap.add_argument(
        "--stability-sampling-seed",
        type=int,
        default=None,
        help="Sampling seed used when --stability-sampling-mode=fixed (defaults to --coarse-seed).",
    )
    ap.add_argument(
        "--stability-val-split-seed",
        type=int,
        default=None,
        help="Validation split seed used when --stability-split-mode=shared_val_seed (defaults to --coarse-seed).",
    )
    ap.add_argument(
        "--strict-fpr-feasibility-check",
        dest="strict_fpr_feasibility_check",
        action="store_true",
        default=True,
        help="Enforce minimum attacked-benign validation denominator before strict FPR gate is considered feasible.",
    )
    ap.add_argument(
        "--no-strict-fpr-feasibility-check",
        dest="strict_fpr_feasibility_check",
        action="store_false",
        help="Disable strict FPR feasibility precheck.",
    )
    ap.add_argument(
        "--min-val-benign-for-fpr-gate",
        type=int,
        default=None,
        help="Minimum attacked-benign validation rows required for strict FPR gate (default ceil(1/gate_clean_fpr_max)).",
    )
    ap.add_argument("--stage2-topk-global", type=int, default=3)
    ap.add_argument("--out-root", type=str, default="reports")
    ap.add_argument("--chunk-size", type=int, default=300000)
    ap.add_argument(
        "--sampling-policy",
        choices=("fixed_count", "adaptive_fraction"),
        default="fixed_count",
        help="Sampling policy for class buckets: fixed counts or adaptive fraction with min/max clamps.",
    )

    ap.add_argument("--hardneg-train-benign-sample", type=int, default=12000)
    ap.add_argument("--hardneg-train-malicious-sample", type=int, default=12000)
    ap.add_argument("--hardneg-val-benign-sample", type=int, default=4000)
    ap.add_argument("--hardneg-train-benign-fraction", type=float, default=0.20)
    ap.add_argument("--hardneg-train-benign-min", type=int, default=4000)
    ap.add_argument("--hardneg-train-benign-max", type=int, default=30000)
    ap.add_argument("--hardneg-train-malicious-fraction", type=float, default=0.08)
    ap.add_argument("--hardneg-train-malicious-min", type=int, default=4000)
    ap.add_argument("--hardneg-train-malicious-max", type=int, default=20000)
    ap.add_argument("--hardneg-val-benign-fraction", type=float, default=0.20)
    ap.add_argument("--hardneg-val-benign-min", type=int, default=3000)
    ap.add_argument("--hardneg-val-benign-max", type=int, default=10000)
    ap.add_argument("--hardneg-epsilons", type=str, default="0.05,0.10")
    ap.add_argument("--hardneg-query-budget", type=int, default=120)
    ap.add_argument("--hardneg-query-max-steps", type=int, default=60)
    ap.add_argument("--hardneg-candidates-per-step", type=int, default=3)
    ap.add_argument("--hardneg-feature-subset-size", type=int, default=3)
    ap.add_argument(
        "--hardneg-query-max-active-rows-per-step",
        type=int,
        default=0,
        help="Optional cap on active rows processed per hillclimb step (<=0 disables cap).",
    )
    ap.add_argument(
        "--hardneg-query-stagnation-patience",
        type=int,
        default=0,
        help="Optional per-attack early-stop patience in hillclimb steps (<=0 disables).",
    )
    ap.add_argument(
        "--hardneg-query-stagnation-min-delta",
        type=float,
        default=1e-6,
        help="Minimum best-margin improvement per step before counting stagnation.",
    )
    ap.add_argument(
        "--query-score-batch-rows",
        type=int,
        default=64,
        help="Rows processed together in each attack scoring batch (higher values increase GPU utilization).",
    )
    ap.add_argument(
        "--query-fast-projection",
        dest="query_fast_projection",
        action="store_true",
        default=True,
        help="Enable fast batched projection with top-k full-refine scoring in hillclimb loops.",
    )
    ap.add_argument(
        "--no-query-fast-projection",
        dest="query_fast_projection",
        action="store_false",
        help="Disable fast projection path and use full projection on all candidate proposals.",
    )
    ap.add_argument(
        "--query-refine-topk",
        type=int,
        default=2,
        help="Top-k fast candidates per row to fully project+rescore in hillclimb loops.",
    )
    ap.add_argument(
        "--hardneg-extra-topk-per-epsilon",
        type=int,
        default=0,
        help="Per-epsilon extra benign near-miss rows (highest final margin) to include as hard negatives.",
    )
    ap.add_argument("--hardneg-weight", type=float, default=2.0)
    ap.add_argument("--maladv-weight", type=float, default=2.0)
    ap.add_argument(
        "--family-pack",
        choices=("baseline", "bluetooth_recovery", "bluetooth_recovery_fallback"),
        default="baseline",
        help="Family configuration pack used in matrix search.",
    )
    ap.add_argument(
        "--bluetooth-hardneg-max-fraction",
        type=float,
        default=1.0,
        help="Cap bluetooth benign hard-negatives to at most this fraction of sampled bluetooth benign rows.",
    )
    ap.add_argument(
        "--bluetooth-benign-min-train-rows",
        type=int,
        default=8000,
        help="Minimum bluetooth benign train rows after split (deterministically reallocated from val when needed).",
    )
    ap.add_argument(
        "--family-a-hardneg-weight",
        type=float,
        default=1.5,
        help="Family A (FPR-priority): hard-negative sample weight.",
    )
    ap.add_argument(
        "--family-a-maladv-weight",
        type=float,
        default=1.5,
        help="Family A (FPR-priority): malicious-adversarial sample weight.",
    )
    ap.add_argument(
        "--family-b-hardneg-weight",
        type=float,
        default=2.0,
        help="Family B (balanced): hard-negative sample weight.",
    )
    ap.add_argument(
        "--family-b-maladv-weight",
        type=float,
        default=2.0,
        help="Family B (balanced): malicious-adversarial sample weight.",
    )
    ap.add_argument(
        "--family-b-extra-topk-per-epsilon",
        type=int,
        default=500,
        help="Family B (balanced): per-epsilon near-miss additions.",
    )
    ap.add_argument(
        "--family-c-extra-topk-per-epsilon",
        type=int,
        default=2500,
        help="Family C (recall-priority): per-epsilon near-miss additions.",
    )
    ap.add_argument(
        "--family-c-hardneg-weight",
        type=float,
        default=2.5,
        help="Family C (recall-priority): hard-negative sample weight.",
    )
    ap.add_argument(
        "--family-c-maladv-weight",
        type=float,
        default=2.5,
        help="Family C (recall-priority): malicious-adversarial sample weight.",
    )

    ap.add_argument("--threshold-grid-size", type=int, default=400)
    ap.add_argument("--base-threshold-fpr-target", type=float, default=0.01)
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
    ap.add_argument(
        "--gate-clean-fpr-max",
        type=float,
        default=0.005,
        help="Hard acceptance gate for clean FPR.",
    )
    ap.add_argument(
        "--gate-attacked-benign-fpr-max",
        type=float,
        default=0.005,
        help="Hard acceptance gate for attacked-benign FPR.",
    )
    ap.add_argument(
        "--gate-adv-malicious-recall-min",
        type=float,
        default=0.99,
        help="Hard acceptance gate for adversarial malicious recall.",
    )
    ap.add_argument(
        "--gate-epsilon",
        type=float,
        default=0.10,
        help="Worst-case epsilon for hard acceptance gates.",
    )
    ap.add_argument(
        "--threshold-gate-attacked-benign-margin",
        type=float,
        default=1.0,
        help="Internal safety margin multiplier for attacked-benign FPR gate during threshold search.",
    )
    ap.add_argument(
        "--stability-extra-seeds",
        type=str,
        default="43",
        help="Comma-separated additional seeds for winner stability rerun (empty to disable).",
    )
    ap.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument(
        "--val-malicious-sample",
        type=int,
        default=4000,
        help="Validation malicious sample size for adversarial recall calibration.",
    )
    ap.add_argument("--val-malicious-fraction", type=float, default=0.08)
    ap.add_argument("--val-malicious-min", type=int, default=3000)
    ap.add_argument("--val-malicious-max", type=int, default=8000)
    ap.add_argument("--val-malicious-query-budget", type=int, default=None)
    ap.add_argument("--val-malicious-query-max-steps", type=int, default=None)
    ap.add_argument("--val-malicious-candidates-per-step", type=int, default=None)
    ap.add_argument("--val-malicious-feature-subset-size", type=int, default=None)
    ap.add_argument("--val-malicious-score-batch-rows", type=int, default=None)
    ap.add_argument("--val-malicious-query-max-active-rows-per-step", type=int, default=None)
    ap.add_argument("--val-malicious-query-stagnation-patience", type=int, default=None)
    ap.add_argument("--val-malicious-query-stagnation-min-delta", type=float, default=None)
    ap.add_argument("--protocol-max-train-rows", type=int, default=0)
    ap.add_argument("--protocol-max-val-rows", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=8)
    ap.add_argument("--mlp-predict-batch-size", type=int, default=65536)
    ap.add_argument(
        "--save-models",
        dest="save_models",
        action="store_true",
        default=True,
        help="Persist trained candidate model artifacts to each candidate directory (enabled by default).",
    )
    ap.add_argument(
        "--no-save-models",
        dest="save_models",
        action="store_false",
        help="Disable saving trained candidate model artifacts.",
    )
    ap.add_argument("--external-benign-csvs", type=str, default="")
    ap.add_argument("--external-benign-max-rows", type=int, default=0)
    ap.add_argument("--external-benign-overlap-check", action="store_true")

    ap.add_argument("--val-mod", type=int, default=5)
    ap.add_argument("--percentile-lower", type=float, default=1.0)
    ap.add_argument("--percentile-upper", type=float, default=99.0)
    ap.add_argument("--relation-quantile-low", type=float, default=0.01)
    ap.add_argument("--relation-quantile-high", type=float, default=0.99)
    ap.add_argument("--relative-cap-default", type=float, default=0.20)
    ap.add_argument("--relative-cap-rate-group", type=float, default=None)
    ap.add_argument("--relative-cap-size-group", type=float, default=None)
    ap.add_argument("--relative-cap-time-group", type=float, default=None)

    ap.add_argument("--constraint-sample-size", type=int, default=300000)
    ap.add_argument("--preview-max-rows", type=int, default=2000)

                                                                        
    ap.add_argument("--coarse-hardneg-train-benign-sample", type=int, default=None)
    ap.add_argument("--coarse-hardneg-train-malicious-sample", type=int, default=None)
    ap.add_argument("--coarse-hardneg-val-benign-sample", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-sample", type=int, default=None)
    ap.add_argument("--coarse-hardneg-train-benign-fraction", type=float, default=None)
    ap.add_argument("--coarse-hardneg-train-benign-min", type=int, default=None)
    ap.add_argument("--coarse-hardneg-train-benign-max", type=int, default=None)
    ap.add_argument("--coarse-hardneg-train-malicious-fraction", type=float, default=None)
    ap.add_argument("--coarse-hardneg-train-malicious-min", type=int, default=None)
    ap.add_argument("--coarse-hardneg-train-malicious-max", type=int, default=None)
    ap.add_argument("--coarse-hardneg-val-benign-fraction", type=float, default=None)
    ap.add_argument("--coarse-hardneg-val-benign-min", type=int, default=None)
    ap.add_argument("--coarse-hardneg-val-benign-max", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-fraction", type=float, default=None)
    ap.add_argument("--coarse-val-malicious-min", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-max", type=int, default=None)
    ap.add_argument("--coarse-hardneg-query-budget", type=int, default=None)
    ap.add_argument("--coarse-hardneg-query-max-steps", type=int, default=None)
    ap.add_argument("--coarse-hardneg-candidates-per-step", type=int, default=None)
    ap.add_argument("--coarse-query-score-batch-rows", type=int, default=None)
    ap.add_argument("--coarse-query-refine-topk", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-query-budget", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-query-max-steps", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-candidates-per-step", type=int, default=None)
    ap.add_argument("--coarse-val-malicious-score-batch-rows", type=int, default=None)

    ap.add_argument("--stability-hardneg-train-benign-sample", type=int, default=None)
    ap.add_argument("--stability-hardneg-train-malicious-sample", type=int, default=None)
    ap.add_argument("--stability-hardneg-val-benign-sample", type=int, default=None)
    ap.add_argument("--stability-val-malicious-sample", type=int, default=None)
    ap.add_argument("--stability-hardneg-train-benign-fraction", type=float, default=None)
    ap.add_argument("--stability-hardneg-train-benign-min", type=int, default=None)
    ap.add_argument("--stability-hardneg-train-benign-max", type=int, default=None)
    ap.add_argument("--stability-hardneg-train-malicious-fraction", type=float, default=None)
    ap.add_argument("--stability-hardneg-train-malicious-min", type=int, default=None)
    ap.add_argument("--stability-hardneg-train-malicious-max", type=int, default=None)
    ap.add_argument("--stability-hardneg-val-benign-fraction", type=float, default=None)
    ap.add_argument("--stability-hardneg-val-benign-min", type=int, default=None)
    ap.add_argument("--stability-hardneg-val-benign-max", type=int, default=None)
    ap.add_argument("--stability-val-malicious-fraction", type=float, default=None)
    ap.add_argument("--stability-val-malicious-min", type=int, default=None)
    ap.add_argument("--stability-val-malicious-max", type=int, default=None)
    ap.add_argument("--stability-hardneg-query-budget", type=int, default=None)
    ap.add_argument("--stability-hardneg-query-max-steps", type=int, default=None)
    ap.add_argument("--stability-hardneg-candidates-per-step", type=int, default=None)
    ap.add_argument("--stability-query-score-batch-rows", type=int, default=None)
    ap.add_argument("--stability-query-refine-topk", type=int, default=None)
    ap.add_argument("--stability-val-malicious-query-budget", type=int, default=None)
    ap.add_argument("--stability-val-malicious-query-max-steps", type=int, default=None)
    ap.add_argument("--stability-val-malicious-candidates-per-step", type=int, default=None)
    ap.add_argument("--stability-val-malicious-score-batch-rows", type=int, default=None)
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


def _combine_robust_metrics(
    p_adv_benign: np.ndarray,
    p_adv_malicious: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pb = np.asarray(p_adv_benign, dtype=np.float64)
    pm = np.asarray(p_adv_malicious, dtype=np.float64)
    if pb.size <= 0 or pm.size <= 0:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "fpr": float("nan"),
            "n_benign": int(pb.size),
            "n_malicious": int(pm.size),
            "n_total": int(pb.size + pm.size),
        }
    y = np.concatenate(
        [
            np.zeros(pb.size, dtype=np.int8),
            np.ones(pm.size, dtype=np.int8),
        ],
        axis=0,
    )
    p = np.concatenate([pb, pm], axis=0)
    pred = (p >= float(threshold)).astype(np.int8, copy=False)
    m = _binary_metrics(y, pred)
    m.update(
        {
            "n_benign": int(pb.size),
            "n_malicious": int(pm.size),
            "n_total": int(pb.size + pm.size),
        }
    )
    return m


def _gate_failure_category(
    *,
    gate_clean_fpr: bool,
    gate_attacked_benign_fpr: bool,
    gate_adv_malicious_recall: bool,
) -> str:
    fpr_fail = (not gate_clean_fpr) or (not gate_attacked_benign_fpr)
    adv_fail = not gate_adv_malicious_recall
    if fpr_fail and adv_fail:
        return "both"
    if fpr_fail:
        return "fpr_only"
    if adv_fail:
        return "adv_recall_only"
    return "none"


def _select_threshold_gate_lexicographic(
    *,
    y_clean: np.ndarray,
    p_clean: np.ndarray,
    p_adv_benign: np.ndarray,
    p_adv_malicious: np.ndarray,
    grid_size: int,
    base_threshold: float,
    gate_clean_fpr_max: float,
    gate_attacked_benign_fpr_max: float,
    gate_adv_malicious_recall_min: float,
    gate_attacked_benign_margin: float = 1.0,
) -> Dict[str, Any]:
    y = np.asarray(y_clean, dtype=np.int8)
    p = np.asarray(p_clean, dtype=np.float64)
    p_adv_b = np.asarray(p_adv_benign, dtype=np.float64)
    p_adv_m = np.asarray(p_adv_malicious, dtype=np.float64)

    all_scores = p[np.isfinite(p)]
    if p_adv_b.size > 0:
        all_scores = np.concatenate([all_scores, p_adv_b[np.isfinite(p_adv_b)]], axis=0)
    if p_adv_m.size > 0:
        all_scores = np.concatenate([all_scores, p_adv_m[np.isfinite(p_adv_m)]], axis=0)
    if all_scores.size <= 0:
        all_scores = np.array([float(base_threshold)], dtype=np.float64)

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

    rows: List[Dict[str, Any]] = []
    has_adv_mal = p_adv_m.size > 0
    margin = float(max(0.0, min(1.0, float(gate_attacked_benign_margin))))
    internal_attacked_benign_target = float(gate_attacked_benign_fpr_max) * margin
    for thr in candidates:
        pred = (p >= thr).astype(np.int8, copy=False)
        clean = _binary_metrics(y, pred)
        attacked_benign_fpr = (
            float(np.mean((p_adv_b >= thr).astype(np.float64)))
            if p_adv_b.size > 0
            else float("nan")
        )
        attacked_malicious_recall = (
            float(np.mean((p_adv_m >= thr).astype(np.float64)))
            if has_adv_mal
            else float("nan")
        )
        gate_clean = bool(float(clean["fpr"]) <= float(gate_clean_fpr_max))
        gate_attacked = bool(
            np.isfinite(attacked_benign_fpr) and float(attacked_benign_fpr) <= float(gate_attacked_benign_fpr_max)
        )
        gate_attacked_internal = bool(
            np.isfinite(attacked_benign_fpr) and float(attacked_benign_fpr) <= float(internal_attacked_benign_target)
        )
        gate_adv = bool(
            np.isfinite(attacked_malicious_recall)
            and float(attacked_malicious_recall) >= float(gate_adv_malicious_recall_min)
        )
        row = {
            "threshold": float(thr),
            "clean_f1": float(clean["f1"]),
            "clean_recall": float(clean["recall"]),
            "clean_precision": float(clean["precision"]),
            "clean_fpr": float(clean["fpr"]),
            "attacked_benign_fpr": float(attacked_benign_fpr),
            "attacked_malicious_recall": float(attacked_malicious_recall),
            "adv_shortfall": (
                max(0.0, float(gate_adv_malicious_recall_min) - float(attacked_malicious_recall))
                if np.isfinite(attacked_malicious_recall)
                else float("inf")
            ),
            "gate_attacked_benign_fpr_internal_target": float(internal_attacked_benign_target),
            "gate_attacked_benign_fpr_internal": gate_attacked_internal,
            "gate_clean_fpr": gate_clean,
            "gate_attacked_benign_fpr": gate_attacked,
            "gate_adv_malicious_recall": gate_adv,
            "gate_pass_internal": bool(gate_clean and gate_attacked_internal and gate_adv),
            "gate_pass": bool(gate_clean and gate_attacked and gate_adv),
            "gate_failure_category": _gate_failure_category(
                gate_clean_fpr=gate_clean,
                gate_attacked_benign_fpr=gate_attacked,
                gate_adv_malicious_recall=gate_adv,
            ),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No threshold candidates were generated.")

    num_internal_gate_pass = int(sum(1 for r in rows if bool(r.get("gate_pass_internal", False))))
    num_gate_pass = int(sum(1 for r in rows if bool(r["gate_pass"])))
    if num_internal_gate_pass > 0:
        best = sorted(
            rows,
            key=lambda r: (
                0 if bool(r.get("gate_pass_internal", False)) else 1,
                float(r["attacked_benign_fpr"]) if np.isfinite(r["attacked_benign_fpr"]) else float("inf"),
                float(r["clean_fpr"]) if np.isfinite(r["clean_fpr"]) else float("inf"),
                -(float(r["attacked_malicious_recall"]) if np.isfinite(r["attacked_malicious_recall"]) else -float("inf")),
                -(float(r["clean_f1"]) if np.isfinite(r["clean_f1"]) else -float("inf")),
                -float(r["threshold"]),
            ),
        )[0]
        selection_mode = "internal_margin_gate_pass_lexicographic"
        selection_rule = (
            "lexicographic_internal_margin_gate_ranking:"
            "gate_pass_internal_margin_then_min_attacked_benign_fpr_then_min_clean_fpr_"
            "then_max_attacked_malicious_recall_then_max_clean_f1"
        )
    elif num_gate_pass > 0:
        best = sorted(
            rows,
            key=lambda r: (
                0 if bool(r["gate_pass"]) else 1,
                float(r["attacked_benign_fpr"]) if np.isfinite(r["attacked_benign_fpr"]) else float("inf"),
                float(r["clean_fpr"]) if np.isfinite(r["clean_fpr"]) else float("inf"),
                -(float(r["attacked_malicious_recall"]) if np.isfinite(r["attacked_malicious_recall"]) else -float("inf")),
                -(float(r["clean_f1"]) if np.isfinite(r["clean_f1"]) else -float("inf")),
                -float(r["threshold"]),
            ),
        )[0]
        selection_mode = "gate_pass_lexicographic"
        selection_rule = (
            "lexicographic_gate_ranking:"
            "gate_pass_then_min_attacked_benign_fpr_then_min_clean_fpr_"
            "then_max_attacked_malicious_recall_then_max_clean_f1"
        )
    else:
        best = sorted(
            rows,
            key=lambda r: (
                float(r["adv_shortfall"]) if np.isfinite(r["adv_shortfall"]) else float("inf"),
                float(r["attacked_benign_fpr"]) if np.isfinite(r["attacked_benign_fpr"]) else float("inf"),
                float(r["clean_fpr"]) if np.isfinite(r["clean_fpr"]) else float("inf"),
                -(float(r["clean_f1"]) if np.isfinite(r["clean_f1"]) else -float("inf")),
                -float(r["threshold"]),
            ),
        )[0]
        selection_mode = "no_pass_adv_priority"
        selection_rule = (
            "fallback_no_gate_pass:"
            "min_adv_shortfall_then_min_attacked_benign_fpr_then_min_clean_fpr_"
            "then_max_clean_f1"
        )

    return {
        "selected_threshold": float(best["threshold"]),
        "selection_rule": selection_rule,
        "selection_mode": selection_mode,
        "selected_metrics": best,
        "selected_adv_shortfall": float(best["adv_shortfall"]),
        "num_candidates": int(candidates.size),
        "num_internal_gate_pass": int(num_internal_gate_pass),
        "num_gate_pass": int(num_gate_pass),
        "gate_attacked_benign_fpr_internal_target": float(internal_attacked_benign_target),
        "gate_attacked_benign_margin": float(margin),
        "gate_clean_fpr_max": float(gate_clean_fpr_max),
        "gate_attacked_benign_fpr_max": float(gate_attacked_benign_fpr_max),
        "gate_adv_malicious_recall_min": float(gate_adv_malicious_recall_min),
    }


def _family_matrix_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    baseline = [
        {
            "family_id": "A",
            "family_name": "fpr_priority",
            "family_description": "strict_flip_only_low_weight",
            "hardneg_weight": float(args.family_a_hardneg_weight),
            "maladv_weight": float(args.family_a_maladv_weight),
            "extra_topk_per_epsilon": 0,
        },
        {
            "family_id": "B",
            "family_name": "balanced",
            "family_description": "strict_flips_limited_near_miss_moderate_weight",
            "hardneg_weight": float(args.family_b_hardneg_weight),
            "maladv_weight": float(args.family_b_maladv_weight),
            "extra_topk_per_epsilon": int(args.family_b_extra_topk_per_epsilon),
        },
        {
            "family_id": "C",
            "family_name": "recall_priority_control",
            "family_description": "current_hardening_control",
            "hardneg_weight": float(args.family_c_hardneg_weight),
            "maladv_weight": float(args.family_c_maladv_weight),
            "extra_topk_per_epsilon": int(args.family_c_extra_topk_per_epsilon),
        },
    ]
    family_pack = str(args.family_pack)
    if family_pack == "baseline":
        return baseline

    if family_pack == "bluetooth_recovery_fallback":
        c_desc = "bluetooth_recovery_fallback_tuned"
        c_hardneg = 1.9
        c_maladv = 2.1
        c_extra_topk = 500
    else:
        c_desc = "bluetooth_recovery_tuned"
        c_hardneg = 2.0
        c_maladv = 2.25
        c_extra_topk = 800

    return [
        baseline[0],
        baseline[1],
        {
            "family_id": "C",
            "family_name": "recall_priority_control",
            "family_description": c_desc,
            "hardneg_weight": c_hardneg,
            "maladv_weight": c_maladv,
            "extra_topk_per_epsilon": c_extra_topk,
        },
        {
            "family_id": "D",
            "family_name": "bluetooth_recovery_d",
            "family_description": "bluetooth_recovery_conservative_plus",
            "hardneg_weight": 1.75,
            "maladv_weight": 2.25,
            "extra_topk_per_epsilon": 250,
        },
        {
            "family_id": "E",
            "family_name": "bluetooth_recovery_e",
            "family_description": "bluetooth_recovery_aggressive_recall",
            "hardneg_weight": 2.25,
            "maladv_weight": 2.75,
            "extra_topk_per_epsilon": 500,
        },
    ]

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


def _apply_bluetooth_benign_train_floor(
    *,
    df: pd.DataFrame,
    val_mask: np.ndarray,
    min_train_rows: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    floor_rows = max(0, int(min_train_rows))
    proto = df["protocol_hint"].fillna("unknown").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
    bt_b_idx = np.flatnonzero((proto == "bluetooth") & (y == 0))
    if bt_b_idx.size <= 0:
        return np.asarray(val_mask, dtype=bool), None

    val = np.asarray(val_mask, dtype=bool).copy()
    bt_b_val_idx = bt_b_idx[val[bt_b_idx]]
    bt_b_train_idx = bt_b_idx[~val[bt_b_idx]]
    train_before = int(bt_b_train_idx.size)
    val_before = int(bt_b_val_idx.size)
    rows_needed = max(0, int(floor_rows) - train_before)
    min_val_rows_kept = 1
    max_movable = max(0, val_before - min_val_rows_kept)
    move_target = min(rows_needed, max_movable)
    moved_rows = 0

    if move_target > 0:
        rel = df["source_relpath"].fillna("").astype(str).to_numpy(dtype=object, copy=False)
        row_idx = pd.to_numeric(df["source_row_index"], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)
        val_rel = rel[bt_b_val_idx]
        unique_files, inv = np.unique(val_rel, return_inverse=True)
        counts = np.bincount(inv, minlength=int(unique_files.size)).astype(np.int64, copy=False)
        file_hash = pd.util.hash_pandas_object(pd.Series(unique_files, dtype="string"), index=False).to_numpy(np.uint64)
        file_order = np.lexsort((file_hash, -counts), axis=-1)

        mix_const = np.uint64(11400714819323198485)
        seed_u64 = np.uint64(int(seed) & 0xFFFFFFFF)
        move_parts: List[np.ndarray] = []
        remaining = int(move_target)
        for file_pos in file_order.tolist():
            if remaining <= 0:
                break
            file_rows = bt_b_val_idx[inv == int(file_pos)]
            if file_rows.size <= 0:
                continue
            rel_hash_rows = pd.util.hash_pandas_object(
                pd.Series(rel[file_rows], dtype="string"),
                index=False,
            ).to_numpy(np.uint64)
            row_key = (np.asarray(row_idx[file_rows], dtype=np.uint64) * mix_const) ^ seed_u64 ^ rel_hash_rows
            row_order = np.argsort(row_key, kind="mergesort")
            take = min(remaining, int(file_rows.size))
            move_parts.append(file_rows[row_order[:take]].astype(np.int64, copy=False))
            remaining -= take
        if move_parts:
            move_idx = np.concatenate(move_parts, axis=0).astype(np.int64, copy=False)
            val[move_idx] = False
            moved_rows = int(move_idx.size)

    bt_b_val_after = int(np.sum(val[bt_b_idx]))
    bt_b_train_after = int(bt_b_idx.size - bt_b_val_after)
    all_rel = df.loc[bt_b_idx, "source_relpath"].fillna("").astype(str).to_numpy(dtype=object, copy=False)
    _, all_counts = np.unique(all_rel, return_counts=True)
    event = {
        "protocol_hint": "bluetooth",
        "label": 0,
        "stratum_rows": int(bt_b_idx.size),
        "stratum_files": int(all_counts.size),
        "fallback_type": "bluetooth_benign_min_train_floor",
        "floor_target": int(floor_rows),
        "floor_applied": bool(moved_rows > 0),
        "rows_moved_to_train": int(moved_rows),
        "train_rows_before_floor": int(train_before),
        "train_rows_after_floor": int(bt_b_train_after),
        "val_rows_before_floor": int(val_before),
        "val_rows_after_floor": int(bt_b_val_after),
        "min_val_rows_kept": int(min_val_rows_kept),
    }
    return val.astype(bool, copy=False), event


def _resolve_stability_split_seed(
    *,
    split_mode: str,
    shared_split_seed: int,
    stability_seed: int,
) -> int:
    mode = str(split_mode).strip().lower()
    if mode == "shared_val_seed":
        return int(shared_split_seed)
    if mode == "per_seed":
        return int(stability_seed)
    raise RuntimeError(f"Unsupported stability split mode: {split_mode}")


def _resolve_stability_sampling_seed(
    *,
    sampling_mode: str,
    shared_sampling_seed: int,
    stability_seed: int,
    protocol: str,
) -> int:
    mode = str(sampling_mode).strip().lower()
    proto = protocol_slug(protocol)
    if mode == "fixed":
        return int(shared_sampling_seed)
    if mode == "per_seed":
        return int(stability_seed)
    if mode == "hybrid":
        if proto == "bluetooth":
            return int(shared_sampling_seed)
        return int(stability_seed)
    raise RuntimeError(f"Unsupported stability sampling mode: {sampling_mode}")


def _compute_fpr_feasibility(
    *,
    val_benign_count: int,
    min_required: int,
    enabled: bool,
) -> Dict[str, Any]:
    count = int(val_benign_count)
    min_req = int(max(1, int(min_required)))
    resolution = (1.0 / float(count)) if count > 0 else float("inf")
    feasible = (not bool(enabled)) or (count >= min_req)
    return {
        "val_benign_count": int(count),
        "min_required": int(min_req),
        "fpr_resolution": float(resolution),
        "strict_fpr_feasible": bool(feasible),
        "strict_fpr_feasibility_check": bool(enabled),
    }


def _apply_data_cap_gate_override(
    selected_metrics: Dict[str, Any],
    *,
    strict_fpr_feasible: bool,
) -> Dict[str, Any]:
    out = dict(selected_metrics)
    if bool(strict_fpr_feasible):
        return out
    out["gate_pass"] = False
    out["gate_failure_category"] = "data_cap_infeasible"
    return out


def _build_data_cap_summary_rows(
    *,
    df: pd.DataFrame,
    val_mask: np.ndarray,
    protocols: Sequence[str],
    fallback_events: Sequence[Dict[str, Any]],
    split_seed_used: int,
) -> List[Dict[str, Any]]:
    proto = df["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
    y = normalize_label_series(df["label"])
    rel = df["source_relpath"].fillna("").astype(str).to_numpy(dtype=object, copy=False)
    val = np.asarray(val_mask, dtype=bool)

    fallback_map: Dict[Tuple[str, int], str] = {}
    floor_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for ev in fallback_events:
        key = (protocol_slug(str(ev.get("protocol_hint", ""))), int(ev.get("label", -1)))
        fallback_type = str(ev.get("fallback_type", "none"))
        if fallback_type == "bluetooth_benign_min_train_floor":
            floor_map[key] = {
                "floor_target": int(ev.get("floor_target", 0)),
                "floor_applied": bool(ev.get("floor_applied", False)),
                "rows_moved_to_train": int(ev.get("rows_moved_to_train", 0)),
                "train_rows_before_floor": int(ev.get("train_rows_before_floor", 0)),
                "train_rows_after_floor": int(ev.get("train_rows_after_floor", 0)),
                "val_rows_before_floor": int(ev.get("val_rows_before_floor", 0)),
                "val_rows_after_floor": int(ev.get("val_rows_after_floor", 0)),
            }
        else:
            fallback_map[key] = fallback_type

    rows: List[Dict[str, Any]] = []
    for p in protocols:
        pp = protocol_slug(p)
        p_mask = proto == pp
        for cls in (0, 1):
            idx = np.flatnonzero(p_mask & (y == int(cls)))
            n_rows = int(idx.size)
            if n_rows <= 0:
                floor_info = floor_map.get((pp, int(cls)), None)
                rows.append(
                    {
                        "protocol": pp,
                        "label": int(cls),
                        "rows": 0,
                        "unique_files": 0,
                        "largest_file_share": float("nan"),
                        "fallback_type": str(fallback_map.get((pp, int(cls)), "none")),
                        "val_rows": 0,
                        "fpr_resolution": float("inf"),
                        "split_seed_used": int(split_seed_used),
                        "floor_target": int(floor_info["floor_target"]) if floor_info else 0,
                        "floor_applied": bool(floor_info["floor_applied"]) if floor_info else False,
                        "rows_moved_to_train": int(floor_info["rows_moved_to_train"]) if floor_info else 0,
                        "benign_train_before_floor": int(floor_info["train_rows_before_floor"]) if floor_info else 0,
                        "benign_train_after_floor": int(floor_info["train_rows_after_floor"]) if floor_info else 0,
                    }
                )
                continue

            rel_sub = rel[idx]
            _, counts = np.unique(rel_sub, return_counts=True)
            unique_files = int(counts.size)
            largest_share = float(np.max(counts) / float(n_rows)) if unique_files > 0 else float("nan")
            val_rows = int(np.sum(val[idx]))
            resolution = (1.0 / float(val_rows)) if val_rows > 0 else float("inf")
            floor_info = floor_map.get((pp, int(cls)), None)
            rows.append(
                {
                    "protocol": pp,
                    "label": int(cls),
                    "rows": int(n_rows),
                    "unique_files": int(unique_files),
                    "largest_file_share": float(largest_share),
                    "fallback_type": str(fallback_map.get((pp, int(cls)), "none")),
                    "val_rows": int(val_rows),
                    "fpr_resolution": float(resolution),
                    "split_seed_used": int(split_seed_used),
                    "floor_target": int(floor_info["floor_target"]) if floor_info else 0,
                    "floor_applied": bool(floor_info["floor_applied"]) if floor_info else False,
                    "rows_moved_to_train": int(floor_info["rows_moved_to_train"]) if floor_info else 0,
                    "benign_train_before_floor": int(floor_info["train_rows_before_floor"]) if floor_info else 0,
                    "benign_train_after_floor": int(floor_info["train_rows_after_floor"]) if floor_info else 0,
                }
            )
    return rows


def _prepare_protocol_split_bundle(
    *,
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    protocols: Sequence[str],
    split_seed: int,
    val_mod: int,
    bluetooth_benign_min_train_rows: int,
    protocol_max_train_rows: int,
    protocol_max_val_rows: int,
    start_ts: float,
) -> Dict[str, Any]:
    val_mask, fallback_events = _build_protocol_class_val_mask(
        train_df[["source_relpath", "source_row_index", "protocol_hint", "label"]],
        seed=int(split_seed),
        val_mod=int(val_mod),
        allow_row_fallback=True,
    )
    val_mask, bt_floor_event = _apply_bluetooth_benign_train_floor(
        df=train_df[["source_relpath", "source_row_index", "protocol_hint", "label"]],
        val_mask=val_mask,
        min_train_rows=int(bluetooth_benign_min_train_rows),
        seed=int(split_seed),
    )
    if bt_floor_event is not None:
        fallback_events.append(bt_floor_event)
    fallback_events_tagged = [dict(ev, split_seed_used=int(split_seed)) for ev in fallback_events]

    proto_all = train_df["protocol_hint"].astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
    x_all = train_df.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, copy=False)
    y_all = normalize_label_series(train_df["label"])
    src_relpath_all = train_df["source_relpath"].astype(str).to_numpy(dtype=object, copy=False)
    src_row_idx_all = pd.to_numeric(train_df["source_row_index"], errors="coerce").fillna(0).astype(np.int64).to_numpy(copy=False)

    train_mask = ~val_mask
    x_train_all = x_all[train_mask]
    y_train_all = y_all[train_mask]
    proto_train_all = proto_all[train_mask]
    src_train_relpath_all = src_relpath_all[train_mask]
    src_train_row_idx_all = src_row_idx_all[train_mask]

    x_val_all = x_all[val_mask]
    y_val_all = y_all[val_mask]
    proto_val_all = proto_all[val_mask]
    src_val_relpath_all = src_relpath_all[val_mask]
    src_val_row_idx_all = src_row_idx_all[val_mask]

    proto_data: Dict[str, Dict[str, Any]] = {}
    for proto in protocols:
        tr_mask = proto_train_all == proto
        va_mask = proto_val_all == proto
        x_tr = x_train_all[tr_mask].astype(np.float32, copy=False)
        y_tr = y_train_all[tr_mask].astype(np.int8, copy=False)
        r_tr = src_train_relpath_all[tr_mask]
        i_tr = src_train_row_idx_all[tr_mask]
        x_va = x_val_all[va_mask].astype(np.float32, copy=False)
        y_va = y_val_all[va_mask].astype(np.int8, copy=False)
        r_va = src_val_relpath_all[va_mask]
        i_va = src_val_row_idx_all[va_mask]
        if x_tr.shape[0] <= 1 or x_va.shape[0] <= 1:
            raise RuntimeError(
                f"Insufficient rows for protocol={proto} under split_seed={split_seed}: "
                f"train={x_tr.shape[0]} val={x_va.shape[0]}"
            )

        proto_seed_offset = _stable_protocol_offset(proto)
        if int(protocol_max_train_rows) > 0 and x_tr.shape[0] > int(protocol_max_train_rows):
            keep = _stratified_subsample_indices(y_tr, int(protocol_max_train_rows), seed=int(split_seed) + proto_seed_offset)
            x_tr, y_tr, r_tr, i_tr = x_tr[keep], y_tr[keep], r_tr[keep], i_tr[keep]
        if int(protocol_max_val_rows) > 0 and x_va.shape[0] > int(protocol_max_val_rows):
            keep = _stratified_subsample_indices(y_va, int(protocol_max_val_rows), seed=int(split_seed) + 17 + proto_seed_offset)
            x_va, y_va, r_va, i_va = x_va[keep], y_va[keep], r_va[keep], i_va[keep]

        tr_b_all = np.flatnonzero(y_tr == 0)
        tr_m_all = np.flatnonzero(y_tr == 1)
        va_b_all = np.flatnonzero(y_va == 0)
        va_m_all = np.flatnonzero(y_va == 1)
        if tr_b_all.size <= 0 or tr_m_all.size <= 0 or va_b_all.size <= 0 or va_m_all.size <= 0:
            raise RuntimeError(
                f"Need both classes for protocol={proto} under split_seed={split_seed}; "
                f"train_b={tr_b_all.size} train_m={tr_m_all.size} val_b={va_b_all.size} val_m={va_m_all.size}"
            )

        proto_data[proto] = {
            "x_train": x_tr,
            "y_train": y_tr,
            "src_train_relpath": r_tr,
            "src_train_row_index": i_tr,
            "x_val": x_va,
            "y_val": y_va,
            "src_val_relpath": r_va,
            "src_val_row_index": i_va,
            "train_benign_all_idx": tr_b_all.astype(np.int64, copy=False),
            "train_malicious_all_idx": tr_m_all.astype(np.int64, copy=False),
            "val_benign_all_idx": va_b_all.astype(np.int64, copy=False),
            "val_malicious_all_idx": va_m_all.astype(np.int64, copy=False),
            "split_seed_used": int(split_seed),
        }

    data_cap_rows = _build_data_cap_summary_rows(
        df=train_df,
        val_mask=val_mask,
        protocols=protocols,
        fallback_events=fallback_events_tagged,
        split_seed_used=int(split_seed),
    )
    log_progress(
        f"split prepared: split_seed={split_seed} fallbacks={len(fallback_events_tagged)}",
        start_ts=start_ts,
    )
    return {
        "split_seed_used": int(split_seed),
        "val_mask": val_mask,
        "fallback_events": fallback_events_tagged,
        "proto_data": proto_data,
        "data_cap_rows": data_cap_rows,
    }


def _sample_protocol_indices_for_run(
    *,
    proto: str,
    proto_data: Dict[str, Any],
    run_seed: int,
    hardneg_train_benign_sample: int,
    hardneg_train_malicious_sample: int,
    hardneg_val_benign_sample: int,
    val_malicious_sample: int,
    sampling_seed: Optional[int] = None,
    sampling_policy: str = "fixed_count",
    sampling_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, np.ndarray]:
    seed_driver = int(sampling_seed) if sampling_seed is not None else int(run_seed)
    seed_base = int(seed_driver) * 10_003 + _stable_protocol_offset(proto)
    rng = np.random.default_rng(seed_base)

    tr_b_all = np.asarray(proto_data["train_benign_all_idx"], dtype=np.int64)
    tr_m_all = np.asarray(proto_data["train_malicious_all_idx"], dtype=np.int64)
    va_b_all = np.asarray(proto_data["val_benign_all_idx"], dtype=np.int64)
    va_m_all = np.asarray(proto_data["val_malicious_all_idx"], dtype=np.int64)

    policy = str(sampling_policy).strip().lower()
    if policy not in {"fixed_count", "adaptive_fraction"}:
        raise RuntimeError(f"Unsupported sampling_policy: {sampling_policy}")
    bucket_cfg = sampling_buckets or {}

    def _want(
        *,
        pool_size: int,
        fixed_count: int,
        bucket_name: str,
    ) -> int:
        if policy != "adaptive_fraction":
            return int(max(0, int(fixed_count)))
        cfg = bucket_cfg.get(bucket_name, {})
        return _resolve_adaptive_sample_target(
            pool_size=int(pool_size),
            fraction=float(cfg.get("fraction", 1.0)),
            min_rows=_optional_nonnegative_int(cfg.get("min_rows", None)),
            max_rows=_optional_nonnegative_int(cfg.get("max_rows", None)),
        )

    tr_b_want = _want(
        pool_size=int(tr_b_all.size),
        fixed_count=int(hardneg_train_benign_sample),
        bucket_name="train_benign",
    )
    tr_m_want = _want(
        pool_size=int(tr_m_all.size),
        fixed_count=int(hardneg_train_malicious_sample),
        bucket_name="train_malicious",
    )
    va_b_want = _want(
        pool_size=int(va_b_all.size),
        fixed_count=int(hardneg_val_benign_sample),
        bucket_name="val_benign",
    )
    va_m_want = _want(
        pool_size=int(va_m_all.size),
        fixed_count=int(val_malicious_sample),
        bucket_name="val_malicious",
    )

    tr_b_pick = _sample_indices(int(tr_b_all.size), int(tr_b_want), rng)
    tr_m_pick = _sample_indices(int(tr_m_all.size), int(tr_m_want), rng)
    va_b_pick = _sample_indices(int(va_b_all.size), int(va_b_want), rng)
    va_m_pick = _sample_indices(int(va_m_all.size), int(va_m_want), rng)

    tr_b = tr_b_all[tr_b_pick]
    tr_m = tr_m_all[tr_m_pick]
    va_b = va_b_all[va_b_pick]
    va_m = va_m_all[va_m_pick]

    if tr_b.size <= 0 or tr_m.size <= 0 or va_b.size <= 0 or va_m.size <= 0:
        raise RuntimeError(
            f"Sampled class support collapsed for protocol={proto} run_seed={run_seed} sampling_seed={seed_driver}: "
            f"train_b={tr_b.size} train_m={tr_m.size} val_b={va_b.size} val_m={va_m.size}"
        )
    return {
        "train_benign_idx": tr_b.astype(np.int64, copy=False),
        "train_malicious_idx": tr_m.astype(np.int64, copy=False),
        "val_benign_idx": va_b.astype(np.int64, copy=False),
        "val_malicious_idx": va_m.astype(np.int64, copy=False),
    }


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


def _load_protocol_train_rows(
    train_csv: Path,
    feature_columns: List[str],
    protocols: Sequence[str],
    chunk_size: int,
    start_ts: float,
) -> pd.DataFrame:
    allowed = {protocol_slug(p) for p in protocols}
    if not allowed:
        raise RuntimeError("No protocols requested.")

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
    kept_rows = 0
    reader = pd.read_csv(train_csv, usecols=usecols, dtype=dtypes, chunksize=max(1, int(chunk_size)))
    for i, chunk in enumerate(reader, start=1):
        total_rows += int(len(chunk))
        proto = np.array(chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(), dtype=object)
        label = normalize_label_series(chunk["label"])
        mask = np.isin(proto, list(allowed))
        if np.any(mask):
            sub = chunk.loc[mask, ["source_relpath", "source_row_index"] + feature_columns].copy()
            sub["protocol_hint"] = proto[mask]
            sub["label"] = label[mask]
            pieces.append(sub)
            kept_rows += int(len(sub))
        if i % 4 == 0:
            log_progress(
                f"protocol load pass: chunks={i}, scanned_rows={total_rows}, kept_rows={kept_rows}",
                start_ts=start_ts,
            )

    if not pieces:
        raise RuntimeError("No requested protocol rows found in training CSV.")
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


def _stratified_subsample_indices(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    n = int(len(y))
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_ratio = float(len(pos_idx)) / max(1.0, float(n))
    take_pos = int(round(max_rows * pos_ratio))
    take_pos = min(take_pos, len(pos_idx))
    take_neg = max_rows - take_pos
    if take_neg > len(neg_idx):
        take_neg = len(neg_idx)
        take_pos = min(len(pos_idx), max_rows - take_neg)

    pos_take = rng.choice(pos_idx, size=take_pos, replace=False) if take_pos > 0 else np.empty(0, dtype=np.int64)
    neg_take = rng.choice(neg_idx, size=take_neg, replace=False) if take_neg > 0 else np.empty(0, dtype=np.int64)
    out = np.concatenate([pos_take, neg_take]).astype(np.int64, copy=False)
    rng.shuffle(out)
    return out


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
    attack_scorer: AttackScorer,
    threshold: float,
    profile: Dict[str, Any],
    epsilons: Sequence[float],
    query_budget: int,
    query_max_steps: int,
    query_candidates_per_step: int,
    query_feature_subset_size: int,
    query_score_batch_rows: int,
    query_fast_projection: bool = True,
    query_refine_topk: int = 2,
    query_max_active_rows_per_step: Optional[int] = None,
    query_stagnation_patience: Optional[int] = None,
    query_stagnation_min_delta: float = 1e-6,
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

    baseline_margin = attack_scorer.score_margin_fn(x_benign.astype(np.float32, copy=False))
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
            booster=None,
            threshold=float(threshold),
            score_margin_fn=attack_scorer.score_margin_fn,
            profile=profile,
            epsilon=eps,
            objective="maximize",
            query_budget=int(query_budget),
            max_steps=int(query_max_steps),
            candidates_per_step=int(query_candidates_per_step),
            feature_subset_size=int(query_feature_subset_size),
            score_batch_rows=int(query_score_batch_rows),
            fast_projection=bool(query_fast_projection),
            refine_topk=max(1, int(query_refine_topk)),
            max_active_rows_per_step=(
                int(query_max_active_rows_per_step) if query_max_active_rows_per_step is not None else None
            ),
            stagnation_patience=(int(query_stagnation_patience) if query_stagnation_patience is not None else None),
            stagnation_min_delta=float(query_stagnation_min_delta),
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
                "attack_source_mode": str(attack_scorer.source_mode),
                "attack_source_detail": str(attack_scorer.source_detail),
                "queries_total": int(np.sum(queries_used)),
                "queries_mean": float(np.mean(queries_used.astype(np.float64))),
                "queries_p95": float(np.percentile(queries_used.astype(np.float64), 95.0)),
                "accepted_moves_mean": float(attack_res["summary"]["accepted_moves_mean"]),
                "steps_executed": int(attack_res["summary"].get("steps_executed", 0)),
                "steps_with_progress": int(attack_res["summary"].get("steps_with_progress", 0)),
                "active_rows_mean_per_step": float(attack_res["summary"].get("active_rows_mean_per_step", float("nan"))),
                "active_rows_p95_per_step": float(attack_res["summary"].get("active_rows_p95_per_step", float("nan"))),
                "active_rows_max_per_step": int(attack_res["summary"].get("active_rows_max_per_step", 0)),
                "max_active_rows_per_step": int(attack_res["summary"].get("max_active_rows_per_step", 0)),
                "stagnation_patience": int(attack_res["summary"].get("stagnation_patience", 0)),
                "stagnation_min_delta": float(attack_res["summary"].get("stagnation_min_delta", float("nan"))),
                "stagnation_steps": int(attack_res["summary"].get("stagnation_steps", 0)),
                "stagnation_triggered": bool(attack_res["summary"].get("stagnation_triggered", False)),
                "query_fast_projection": bool(attack_res["summary"].get("fast_projection", True)),
                "query_refine_topk": int(attack_res["summary"].get("refine_topk", 1)),
                "fast_projection_rows": int(attack_res["summary"].get("fast_projection_rows", 0)),
                "full_projection_rows": int(attack_res["summary"].get("full_projection_rows", 0)),
                "fast_refine_candidates_scored": int(attack_res["summary"].get("fast_refine_candidates_scored", 0)),
                "full_projection_candidates": int(attack_res["summary"].get("full_projection_candidates", 0)),
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
    attack_scorer: AttackScorer,
    threshold: float,
    profile: Dict[str, Any],
    epsilon: float,
    query_budget: int,
    query_max_steps: int,
    query_candidates_per_step: int,
    query_feature_subset_size: int,
    query_score_batch_rows: int,
    query_fast_projection: bool = True,
    query_refine_topk: int = 2,
    query_max_active_rows_per_step: Optional[int] = None,
    query_stagnation_patience: Optional[int] = None,
    query_stagnation_min_delta: float = 1e-6,
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
                "attack_source_mode": str(attack_scorer.source_mode),
                "attack_source_detail": str(attack_scorer.source_detail),
                "queries_total": 0,
                "queries_mean": float("nan"),
                "queries_p95": float("nan"),
                "accepted_moves_mean": float("nan"),
                "steps_executed": 0,
                "steps_with_progress": 0,
                "active_rows_mean_per_step": float("nan"),
                "active_rows_p95_per_step": float("nan"),
                "active_rows_max_per_step": 0,
                "max_active_rows_per_step": 0,
                "stagnation_patience": 0,
                "stagnation_min_delta": float("nan"),
                "stagnation_steps": 0,
                "stagnation_triggered": False,
                "query_fast_projection": bool(query_fast_projection),
                "query_refine_topk": int(max(1, int(query_refine_topk))),
                "fast_projection_rows": 0,
                "full_projection_rows": 0,
                "fast_refine_candidates_scored": 0,
                "full_projection_candidates": 0,
            },
            "preview_rows": [],
        }

    baseline_margin = attack_scorer.score_margin_fn(x_malicious.astype(np.float32, copy=False))
    baseline_detected = baseline_margin >= 0.0
    rng = np.random.default_rng(int(seed))
    attack_res = run_query_sparse_hillclimb(
        x_orig=x_malicious.astype(np.float32, copy=False),
        baseline_margin=baseline_margin,
        booster=None,
        threshold=float(threshold),
        score_margin_fn=attack_scorer.score_margin_fn,
        profile=profile,
        epsilon=float(epsilon),
        objective="minimize",
        query_budget=int(query_budget),
        max_steps=int(query_max_steps),
        candidates_per_step=int(query_candidates_per_step),
        feature_subset_size=int(query_feature_subset_size),
        score_batch_rows=int(query_score_batch_rows),
        fast_projection=bool(query_fast_projection),
        refine_topk=max(1, int(query_refine_topk)),
        max_active_rows_per_step=(
            int(query_max_active_rows_per_step) if query_max_active_rows_per_step is not None else None
        ),
        stagnation_patience=(int(query_stagnation_patience) if query_stagnation_patience is not None else None),
        stagnation_min_delta=float(query_stagnation_min_delta),
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
        "attack_source_mode": str(attack_scorer.source_mode),
        "attack_source_detail": str(attack_scorer.source_detail),
        "queries_total": int(np.sum(queries_used)),
        "queries_mean": float(np.mean(queries_used.astype(np.float64))),
        "queries_p95": float(np.percentile(queries_used.astype(np.float64), 95.0)),
        "accepted_moves_mean": float(attack_res["summary"]["accepted_moves_mean"]),
        "steps_executed": int(attack_res["summary"].get("steps_executed", 0)),
        "steps_with_progress": int(attack_res["summary"].get("steps_with_progress", 0)),
        "active_rows_mean_per_step": float(attack_res["summary"].get("active_rows_mean_per_step", float("nan"))),
        "active_rows_p95_per_step": float(attack_res["summary"].get("active_rows_p95_per_step", float("nan"))),
        "active_rows_max_per_step": int(attack_res["summary"].get("active_rows_max_per_step", 0)),
        "max_active_rows_per_step": int(attack_res["summary"].get("max_active_rows_per_step", 0)),
        "stagnation_patience": int(attack_res["summary"].get("stagnation_patience", 0)),
        "stagnation_min_delta": float(attack_res["summary"].get("stagnation_min_delta", float("nan"))),
        "stagnation_steps": int(attack_res["summary"].get("stagnation_steps", 0)),
        "stagnation_triggered": bool(attack_res["summary"].get("stagnation_triggered", False)),
        "query_fast_projection": bool(attack_res["summary"].get("fast_projection", True)),
        "query_refine_topk": int(attack_res["summary"].get("refine_topk", 1)),
        "fast_projection_rows": int(attack_res["summary"].get("fast_projection_rows", 0)),
        "full_projection_rows": int(attack_res["summary"].get("full_projection_rows", 0)),
        "fast_refine_candidates_scored": int(attack_res["summary"].get("fast_refine_candidates_scored", 0)),
        "full_projection_candidates": int(attack_res["summary"].get("full_projection_candidates", 0)),
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


@dataclass
class TrainedModelBundle:
    model_name: str
    protocol: str
    model: Any
    predict_proba_fn: Callable[[np.ndarray], np.ndarray]
    trained_device: str
    metadata: Dict[str, Any]


@dataclass
class AttackScorer:
    source_mode: str
    source_detail: str
    threshold: float
    score_margin_fn: Callable[[np.ndarray], np.ndarray]


def _make_xgb_attack_scorer(
    *,
    booster: xgb.Booster,
    threshold: float,
    source_detail: str,
) -> AttackScorer:
    thr = float(threshold)

    def _score(x_batch: np.ndarray) -> np.ndarray:
        return score_margin_batch(booster, x_batch.astype(np.float32, copy=False), threshold=thr)

    return AttackScorer(
        source_mode="fixed_xgb",
        source_detail=str(source_detail),
        threshold=thr,
        score_margin_fn=_score,
    )


def _make_model_attack_scorer(
    *,
    model_bundle: TrainedModelBundle,
    threshold: float,
    source_detail: str,
) -> AttackScorer:
    thr = float(threshold)

    def _score(x_batch: np.ndarray) -> np.ndarray:
        if x_batch.size <= 0:
            return np.empty(0, dtype=np.float64)
        probs = model_bundle.predict_proba_fn(x_batch.astype(np.float32, copy=False))
        return np.asarray(probs, dtype=np.float64) - thr

    return AttackScorer(
        source_mode="candidate_model",
        source_detail=str(source_detail),
        threshold=thr,
        score_margin_fn=_score,
    )


def _resolve_attack_source_mode_for_stage(requested_mode: str, stage_name: str) -> str:
    mode = str(requested_mode).strip().lower()
    stage = str(stage_name).strip().lower()
    if mode not in {"fixed_xgb", "candidate_model", "hybrid"}:
        raise ValueError(f"Unsupported attack source mode: {requested_mode}")
    if mode == "hybrid":
        if stage == "coarse":
            return "fixed_xgb"
        if stage == "stability":
            return "candidate_model"
        raise ValueError(f"Unsupported stage for hybrid attack mode: {stage_name}")
    return mode


def _cap_bluetooth_hardneg_rows(
    *,
    protocol: str,
    x_hardneg: np.ndarray,
    sampled_benign_count: int,
    max_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    proto = protocol_slug(protocol)
    cap_frac = float(max_fraction)
    before_rows = int(x_hardneg.shape[0])
    info = {
        "enabled": bool(proto == "bluetooth" and cap_frac < 1.0),
        "protocol": proto,
        "sampled_benign_count": int(sampled_benign_count),
        "max_fraction": float(cap_frac),
        "before_rows": int(before_rows),
        "after_rows": int(before_rows),
        "capped": False,
        "max_allowed_rows": int(before_rows),
    }
    if proto != "bluetooth" or before_rows <= 0 or cap_frac >= 1.0:
        return x_hardneg, info

    max_allowed = int(math.floor(max(0.0, cap_frac) * float(max(0, int(sampled_benign_count)))))
    info["max_allowed_rows"] = int(max_allowed)
    if before_rows <= max_allowed:
        return x_hardneg, info
    if max_allowed <= 0:
        info["after_rows"] = 0
        info["capped"] = True
        return np.empty((0, x_hardneg.shape[1]), dtype=np.float32), info

    rng = np.random.default_rng(int(seed) + 550_001)
    keep = rng.choice(before_rows, size=max_allowed, replace=False)
    keep.sort()
    capped = x_hardneg[keep].astype(np.float32, copy=False)
    info["after_rows"] = int(capped.shape[0])
    info["capped"] = bool(capped.shape[0] != before_rows)
    return capped, info


if HAS_TORCH:
    class _MLPBinary(nn.Module):                      
        def __init__(self, in_dim: int, width: int, depth: int, dropout: float) -> None:
            super().__init__()
            layers: List[nn.Module] = []                          
            d = int(in_dim)
            for _ in range(max(1, int(depth))):
                layers.append(nn.Linear(d, int(width)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropout)))
                d = int(width)
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x).squeeze(1)
else:
    class _MLPBinary:                      
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("torch is not installed; MLP model is unavailable.")


def _select_base_threshold_from_clean_fpr(
    y_val: np.ndarray,
    p_val: np.ndarray,
    max_fpr: float,
) -> float:
    y = np.asarray(y_val, dtype=np.int8)
    p = np.asarray(p_val, dtype=np.float64)
    if y.size <= 0:
        return 0.5
    cand = np.unique(np.linspace(float(np.min(p)), float(np.max(p)), num=300, dtype=np.float64))
    best_thr = 0.5
    best_obj = -float("inf")
    for thr in cand:
        pred = (p >= thr).astype(np.int8, copy=False)
        m = _binary_metrics(y, pred)
        feasible = float(m["fpr"]) <= float(max_fpr)
        obj = float(m["f1"]) if feasible else float(m["f1"]) - 4.0 * max(0.0, float(m["fpr"]) - float(max_fpr))
        if obj > best_obj:
            best_obj = obj
            best_thr = float(thr)
    return float(best_thr)


def _load_best_hparams(base_run_dir: Path) -> Dict[str, Any]:
    path = base_run_dir / "best_hparams.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def _fit_model_from_scratch(
    *,
    protocol: str,
    model_name: str,
    model_hparams: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    requested_device: str,
    mlp_epochs: int,
    mlp_predict_batch_size: int,
    start_ts: float,
) -> TrainedModelBundle:
    name = str(model_name).strip().lower()
    proto = protocol_slug(protocol)
    thread_count = _runtime_thread_count()

    if name == "xgboost":
        device_attempts = [requested_device] if requested_device in {"cpu", "cuda"} else ["cpu"]
        if requested_device == "cuda":
            device_attempts = ["cuda", "cpu"]

        last_error: Optional[Exception] = None
        for device in device_attempts:
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": device,
                "eval_metric": "aucpr",
                "seed": int(seed),
                "max_depth": int(model_hparams.get("max_depth", 6)),
                "eta": float(model_hparams.get("learning_rate", 0.05)),
                "subsample": float(model_hparams.get("subsample", 0.8)),
                "colsample_bytree": float(model_hparams.get("colsample_bytree", 0.8)),
                "min_child_weight": float(model_hparams.get("min_child_weight", 1.0)),
                "lambda": float(model_hparams.get("reg_lambda", 1.0)),
                "alpha": float(model_hparams.get("reg_alpha", 0.0)),
                "gamma": float(model_hparams.get("gamma", 0.0)),
                "max_bin": int(model_hparams.get("max_bin", 256)),
                "nthread": int(thread_count),
            }
            n_rounds = int(model_hparams.get("n_estimators", 1200))
            try:
                log_progress(f"fit[{proto}:{name}] device={device} nthread={thread_count}", start_ts=start_ts)
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
                    num_boost_round=n_rounds,
                    evals=[(dval, "val")],
                    verbose_eval=False,
                )

                def _pred(x_batch: np.ndarray) -> np.ndarray:
                    if x_batch.size <= 0:
                        return np.empty(0, dtype=np.float64)
                    x32 = x_batch.astype(np.float32, copy=False)
                    try:
                        pred = booster.inplace_predict(x32, validate_features=False)
                        return np.asarray(pred, dtype=np.float64)
                    except Exception:
                        return booster.predict(xgb.DMatrix(x32)).astype(np.float64, copy=False)

                return TrainedModelBundle(
                    model_name=name,
                    protocol=proto,
                    model=booster,
                    predict_proba_fn=_pred,
                    trained_device=device,
                    metadata={"n_rounds": int(n_rounds)},
                )
            except Exception as ex:
                last_error = ex
                if device == "cuda":
                    log_progress(f"fit[{proto}:{name}] cuda failed, retry cpu: {ex}", start_ts=start_ts)
                    continue
                raise
        raise RuntimeError(f"Could not fit xgboost for protocol={proto}: {last_error}")

    if name == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost is not installed in this environment.")
        device_attempts = [requested_device] if requested_device in {"cpu", "cuda"} else ["cpu"]
        if requested_device == "cuda":
            device_attempts = ["cuda", "cpu"]
        last_error = None
        for device in device_attempts:
            task_type = "GPU" if device == "cuda" else "CPU"
            params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "iterations": int(model_hparams.get("iterations", 2000)),
                "depth": int(model_hparams.get("depth", 8)),
                "learning_rate": float(model_hparams.get("learning_rate", 0.05)),
                "l2_leaf_reg": float(model_hparams.get("l2_leaf_reg", 3.0)),
                "random_strength": float(model_hparams.get("random_strength", 1.0)),
                "bagging_temperature": float(model_hparams.get("bagging_temperature", 0.0)),
                "border_count": int(model_hparams.get("border_count", 128)),
                "random_seed": int(seed),
                "verbose": False,
                "task_type": task_type,
                "thread_count": int(thread_count),
            }
            try:
                log_progress(f"fit[{proto}:{name}] device={device} thread_count={thread_count}", start_ts=start_ts)
                model = CatBoostClassifier(**params)
                model.fit(
                    x_train.astype(np.float32, copy=False),
                    y_train.astype(np.int32, copy=False),
                    sample_weight=sample_weight.astype(np.float32, copy=False),
                    eval_set=(x_val.astype(np.float32, copy=False), y_val.astype(np.int32, copy=False)),
                    use_best_model=False,
                    verbose=False,
                )

                def _pred(x_batch: np.ndarray) -> np.ndarray:
                    if x_batch.size <= 0:
                        return np.empty(0, dtype=np.float64)
                    return model.predict_proba(x_batch.astype(np.float32, copy=False))[:, 1].astype(
                        np.float64, copy=False
                    )

                return TrainedModelBundle(
                    model_name=name,
                    protocol=proto,
                    model=model,
                    predict_proba_fn=_pred,
                    trained_device=device,
                    metadata={"iterations": int(params["iterations"])},
                )
            except Exception as ex:
                last_error = ex
                if device == "cuda":
                    log_progress(f"fit[{proto}:{name}] cuda failed, retry cpu: {ex}", start_ts=start_ts)
                    continue
                raise
        raise RuntimeError(f"Could not fit catboost for protocol={proto}: {last_error}")

    if name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is not installed in this environment.")
        params = {
            "objective": "binary",
            "n_estimators": int(model_hparams.get("n_estimators", 1200)),
            "num_leaves": int(model_hparams.get("num_leaves", 127)),
            "max_depth": int(model_hparams.get("max_depth", -1)),
            "learning_rate": float(model_hparams.get("learning_rate", 0.05)),
            "min_child_samples": int(model_hparams.get("min_child_samples", 100)),
            "subsample": float(model_hparams.get("subsample", 0.9)),
            "subsample_freq": int(model_hparams.get("subsample_freq", 1)),
            "colsample_bytree": float(model_hparams.get("colsample_bytree", 0.8)),
            "reg_lambda": float(model_hparams.get("reg_lambda", 1.0)),
            "reg_alpha": float(model_hparams.get("reg_alpha", 0.0)),
            "min_split_gain": float(model_hparams.get("min_split_gain", 0.0)),
            "random_state": int(seed),
            "n_jobs": int(thread_count),
            "verbosity": -1,
        }
        if requested_device == "cuda":
            params["device_type"] = "gpu"
        try:
            log_progress(f"fit[{proto}:{name}] device={params.get('device_type','cpu')}", start_ts=start_ts)
            model = lgb.LGBMClassifier(**params)
            model.fit(
                x_train.astype(np.float32, copy=False),
                y_train.astype(np.int8, copy=False),
                sample_weight=sample_weight.astype(np.float32, copy=False),
            )
        except Exception as ex:
            if requested_device == "cuda":
                log_progress(f"fit[{proto}:{name}] gpu failed, retry cpu: {ex}", start_ts=start_ts)
                params.pop("device_type", None)
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    x_train.astype(np.float32, copy=False),
                    y_train.astype(np.int8, copy=False),
                    sample_weight=sample_weight.astype(np.float32, copy=False),
                )
            else:
                raise

        def _pred(x_batch: np.ndarray) -> np.ndarray:
            if x_batch.size <= 0:
                return np.empty(0, dtype=np.float64)
            return model.predict_proba(x_batch.astype(np.float32, copy=False))[:, 1].astype(np.float64, copy=False)

        return TrainedModelBundle(
            model_name=name,
            protocol=proto,
            model=model,
            predict_proba_fn=_pred,
            trained_device=str(params.get("device_type", "cpu")),
            metadata={"n_estimators": int(params["n_estimators"])},
        )

    if name == "mlp":
        if not HAS_TORCH:
            raise RuntimeError("torch is not installed in this environment.")
        in_dim = int(x_train.shape[1])
        width = int(model_hparams.get("width", 384))
        depth = int(model_hparams.get("depth", 4))
        dropout = float(model_hparams.get("dropout", 0.15))
        lr = float(model_hparams.get("lr", 2e-4))
        weight_decay = float(model_hparams.get("weight_decay", 1e-5))
        batch_size = int(model_hparams.get("batch_size", 8192))
        epochs = max(1, int(mlp_epochs))

        device = "cuda" if (requested_device == "cuda" and torch.cuda.is_available()) else "cpu"
        log_progress(f"fit[{proto}:{name}] device={device}", start_ts=start_ts)

        x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

        x_train_n = ((x_train.astype(np.float32, copy=False) - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        y_train_f = y_train.astype(np.float32, copy=False)
        w_train_f = sample_weight.astype(np.float32, copy=False)

        model = _MLPBinary(in_dim=in_dim, width=width, depth=depth, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)                              
        bce = nn.BCEWithLogitsLoss(reduction="none")                          

        ds = TensorDataset(
            torch.from_numpy(x_train_n),
            torch.from_numpy(y_train_f),
            torch.from_numpy(w_train_f),
        )
        loader = DataLoader(ds, batch_size=max(256, batch_size), shuffle=True, drop_last=False)
        model.train()
        for _ in range(epochs):
            for xb, yb, wb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss_vec = bce(logits, yb)
                loss = (loss_vec * wb).mean()
                loss.backward()
                optimizer.step()

        model.eval()

        def _pred(x_batch: np.ndarray) -> np.ndarray:
            if x_batch.size <= 0:
                return np.empty(0, dtype=np.float64)
            x_n = ((x_batch.astype(np.float32, copy=False) - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
            out: List[np.ndarray] = []
            bs = max(1024, int(mlp_predict_batch_size))
            with torch.no_grad():                              
                for i in range(0, x_n.shape[0], bs):
                    xb = torch.from_numpy(x_n[i : i + bs]).to(device, non_blocking=True)
                    logits = model(xb)
                    prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
                    out.append(prob)
            return np.concatenate(out, axis=0) if out else np.empty(0, dtype=np.float64)

        return TrainedModelBundle(
            model_name=name,
            protocol=proto,
            model=model,
            predict_proba_fn=_pred,
            trained_device=device,
            metadata={
                "epochs": int(epochs),
                "width": int(width),
                "depth": int(depth),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "x_mean": x_mean.astype(np.float32, copy=False),
                "x_std": x_std.astype(np.float32, copy=False),
            },
        )

    raise RuntimeError(f"Unsupported model name: {model_name}")


def _save_model_artifact(bundle: TrainedModelBundle, target_path: Path) -> str:
    name = bundle.model_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        if not str(target_path).endswith(".json"):
            target_path = target_path.with_suffix(".json")
        bundle.model.save_model(str(target_path))
        return str(target_path)
    if name == "catboost":
        if not str(target_path).endswith(".cbm"):
            target_path = target_path.with_suffix(".cbm")
        bundle.model.save_model(str(target_path))
        return str(target_path)
    if name == "lightgbm":
        if not str(target_path).endswith(".txt"):
            target_path = target_path.with_suffix(".txt")
        booster = bundle.model.booster_
        booster.save_model(str(target_path))
        return str(target_path)
    if name == "mlp":
        if not str(target_path).endswith(".pt"):
            target_path = target_path.with_suffix(".pt")
        if not HAS_TORCH:
            raise RuntimeError("torch not available for saving MLP state.")
        payload = {
            "state_dict": bundle.model.state_dict(),
            "metadata": {
                k: v
                for k, v in bundle.metadata.items()
                if k not in {"x_mean", "x_std"}
            },
            "x_mean": bundle.metadata.get("x_mean"),
            "x_std": bundle.metadata.get("x_std"),
        }
        torch.save(payload, str(target_path))                              
        return str(target_path)
    raise RuntimeError(f"Unsupported model for save: {name}")


def _run_single_family(
    *,
    family_cfg: Dict[str, Any],
    run_seed: int,
    start_ts: float,
    out_family_dir: Optional[Path],
    base_run_dir: Path,
    thresholds_by_protocol: Dict[str, float],
    wifi_threshold_base: float,
    feature_columns: List[str],
    wifi_hparams: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    src_train_relpath: np.ndarray,
    src_train_row_index: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    src_val_relpath: np.ndarray,
    src_val_row_index: np.ndarray,
    train_benign_idx: np.ndarray,
    val_benign_idx: np.ndarray,
    val_malicious_idx: np.ndarray,
    wifi_base_booster: xgb.Booster,
    wifi_profile: Dict[str, Any],
    constraints_summary_df: pd.DataFrame,
    realism_profile_json: Dict[str, Any],
    epsilons: Sequence[float],
    gate_epsilon: float,
    query_budget_benign: int,
    query_max_steps_benign: int,
    query_candidates_benign: int,
    query_subset_benign: int,
    query_score_batch_benign: int,
    query_budget_mal: int,
    query_max_steps_mal: int,
    query_candidates_mal: int,
    query_subset_mal: int,
    query_score_batch_mal: int,
    threshold_grid_size: int,
    gate_clean_fpr_max: float,
    gate_attacked_benign_fpr_max: float,
    gate_adv_malicious_recall_min: float,
    xgb_device: str,
    preview_max_rows: int,
) -> Dict[str, Any]:
    family_name = str(family_cfg["family_name"])
    family_id = str(family_cfg["family_id"])
    hardneg_weight = float(family_cfg["hardneg_weight"])
    extra_topk = int(family_cfg["extra_topk_per_epsilon"])

    log_progress(
        (
            f"family[{family_id}:{family_name}] start: seed={int(run_seed)}, "
            f"hardneg_weight={hardneg_weight}, extra_topk_per_epsilon={extra_topk}"
        ),
        start_ts=start_ts,
    )
    fixed_attack_scorer = _make_xgb_attack_scorer(
        booster=wifi_base_booster,
        threshold=float(wifi_threshold_base),
        source_detail="fixed_xgb:wifi_base_booster",
    )

    train_campaign = _run_benign_query_campaign(
        split_name=f"train_{family_name}",
        x_benign=x_train[train_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_train_relpath[train_benign_idx],
        src_row_index=src_train_row_index[train_benign_idx],
        attack_scorer=fixed_attack_scorer,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        query_score_batch_rows=query_score_batch_benign,
        extra_topk_per_epsilon=extra_topk,
        seed=int(run_seed),
        start_ts=start_ts,
    )
    val_campaign = _run_benign_query_campaign(
        split_name=f"val_{family_name}",
        x_benign=x_val[val_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_benign_idx],
        src_row_index=src_val_row_index[val_benign_idx],
        attack_scorer=fixed_attack_scorer,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        query_score_batch_rows=query_score_batch_benign,
        extra_topk_per_epsilon=extra_topk,
        seed=int(run_seed) + 91_001,
        start_ts=start_ts,
    )

    n_hardneg_flips = int(train_campaign["flipped_adv"].shape[0])
    x_hardneg = train_campaign["selected_adv"]
    n_hardneg = int(x_hardneg.shape[0])
    if n_hardneg <= 0:
        raise RuntimeError(
            f"family[{family_name}] produced zero train hard negatives. "
            "Increase benign sample size/query budget or relax constraints."
        )

    x_train_aug = np.vstack([x_train, x_hardneg.astype(np.float32, copy=False)])
    y_train_aug = np.concatenate([y_train, np.zeros(n_hardneg, dtype=np.int8)], axis=0)
    sample_weight = np.ones(y_train_aug.shape[0], dtype=np.float32)
    sample_weight[-n_hardneg:] = hardneg_weight

    wifi_model, trained_device = _fit_wifi_xgb(
        x_train=x_train_aug,
        y_train=y_train_aug,
        sample_weight=sample_weight,
        x_val=x_val,
        y_val=y_val,
        hparams=wifi_hparams,
        seed=int(run_seed),
        requested_device=str(xgb_device),
        start_ts=start_ts,
    )

    p_val_clean = wifi_model.predict(xgb.DMatrix(x_val.astype(np.float32, copy=False))).astype(np.float64, copy=False)

    gate_eps_mask = np.isclose(
        val_campaign["all_adv_eps"].astype(np.float64, copy=False),
        float(gate_epsilon),
        rtol=0.0,
        atol=1e-12,
    )
    if int(np.sum(gate_eps_mask)) <= 0:
        raise RuntimeError(
            f"family[{family_name}] has no attacked benign rows for gate_epsilon={float(gate_epsilon)}."
        )
    x_val_adv_gate = val_campaign["all_adv"][gate_eps_mask].astype(np.float32, copy=False)
    p_val_adv_benign_gate = wifi_model.predict(xgb.DMatrix(x_val_adv_gate)).astype(np.float64, copy=False)

    mal_eval = _run_malicious_query_eval(
        split_name=f"val_{family_name}",
        x_malicious=x_val[val_malicious_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_malicious_idx],
        src_row_index=src_val_row_index[val_malicious_idx],
        attack_scorer=fixed_attack_scorer,
        threshold=wifi_threshold_base,
        profile=wifi_profile,
        epsilon=float(gate_epsilon),
        query_budget=query_budget_mal,
        query_max_steps=query_max_steps_mal,
        query_candidates_per_step=query_candidates_mal,
        query_feature_subset_size=query_subset_mal,
        query_score_batch_rows=query_score_batch_mal,
        seed=int(run_seed) + 303_031,
        start_ts=start_ts,
    )
    p_val_adv_malicious = (
        wifi_model.predict(xgb.DMatrix(mal_eval["x_adv"].astype(np.float32, copy=False))).astype(np.float64, copy=False)
        if mal_eval["x_adv"].shape[0] > 0
        else np.empty(0, dtype=np.float64)
    )

    threshold_sel = _select_threshold_gate_lexicographic(
        y_clean=y_val,
        p_clean=p_val_clean,
        p_adv_benign=p_val_adv_benign_gate,
        p_adv_malicious=p_val_adv_malicious,
        grid_size=int(threshold_grid_size),
        base_threshold=float(wifi_threshold_base),
        gate_clean_fpr_max=float(gate_clean_fpr_max),
        gate_attacked_benign_fpr_max=float(gate_attacked_benign_fpr_max),
        gate_adv_malicious_recall_min=float(gate_adv_malicious_recall_min),
    )
    wifi_threshold_new = float(threshold_sel["selected_threshold"])

    val_clean_pred_new = (p_val_clean >= wifi_threshold_new).astype(np.int8)
    val_clean_pred_old = (p_val_clean >= wifi_threshold_base).astype(np.int8)
    val_clean_metrics_new = _binary_metrics(y_val, val_clean_pred_new)
    val_clean_metrics_old = _binary_metrics(y_val, val_clean_pred_old)

    val_adv_benign_fpr_old = float(np.mean((p_val_adv_benign_gate >= wifi_threshold_base).astype(np.float64)))
    val_adv_benign_fpr_new = float(np.mean((p_val_adv_benign_gate >= wifi_threshold_new).astype(np.float64)))
    val_adv_mal_recall_old = (
        float(np.mean((p_val_adv_malicious >= wifi_threshold_base).astype(np.float64)))
        if p_val_adv_malicious.size > 0
        else float("nan")
    )
    val_adv_mal_recall_new = (
        float(np.mean((p_val_adv_malicious >= wifi_threshold_new).astype(np.float64)))
        if p_val_adv_malicious.size > 0
        else float("nan")
    )

    robust_metrics_old = _combine_robust_metrics(
        p_adv_benign=p_val_adv_benign_gate,
        p_adv_malicious=p_val_adv_malicious,
        threshold=float(wifi_threshold_base),
    )
    robust_metrics_new = _combine_robust_metrics(
        p_adv_benign=p_val_adv_benign_gate,
        p_adv_malicious=p_val_adv_malicious,
        threshold=float(wifi_threshold_new),
    )

    selected_gate = threshold_sel["selected_metrics"]
    gate_pass = bool(selected_gate["gate_pass"])
    gate_failure_category = str(selected_gate["gate_failure_category"])

    out_payload = {
        "family_id": family_id,
        "family_name": family_name,
        "family_description": str(family_cfg["family_description"]),
        "seed": int(run_seed),
        "hardneg_weight": float(hardneg_weight),
        "extra_topk_per_epsilon": int(extra_topk),
        "trained_device": str(trained_device),
        "hard_negative": {
            "train_flip_rows": int(n_hardneg_flips),
            "train_selected_rows": int(n_hardneg),
            "train_campaign_rows": to_jsonable(train_campaign["stats_rows"]),
            "val_campaign_rows": to_jsonable(val_campaign["stats_rows"]),
            "val_malicious_campaign": to_jsonable(mal_eval["stats_row"]),
        },
        "thresholds": {
            "wifi_base": float(wifi_threshold_base),
            "wifi_new": float(wifi_threshold_new),
            "selection": to_jsonable(threshold_sel),
        },
        "validation_metrics": {
            "clean_old_threshold": val_clean_metrics_old,
            "clean_new_threshold": val_clean_metrics_new,
            "attacked_benign_fpr_old_threshold": float(val_adv_benign_fpr_old),
            "attacked_benign_fpr_new_threshold": float(val_adv_benign_fpr_new),
            "attacked_malicious_recall_old_threshold": float(val_adv_mal_recall_old),
            "attacked_malicious_recall_new_threshold": float(val_adv_mal_recall_new),
            "robust_old_threshold": robust_metrics_old,
            "robust_new_threshold": robust_metrics_new,
            "delta_clean_f1": float(val_clean_metrics_new["f1"] - val_clean_metrics_old["f1"]),
            "delta_clean_recall": float(val_clean_metrics_new["recall"] - val_clean_metrics_old["recall"]),
            "delta_robust_f1": float(robust_metrics_new["f1"] - robust_metrics_old["f1"])
            if np.isfinite(robust_metrics_old["f1"]) and np.isfinite(robust_metrics_new["f1"])
            else float("nan"),
        },
        "gates": {
            "gate_clean_fpr_max": float(gate_clean_fpr_max),
            "gate_attacked_benign_fpr_max": float(gate_attacked_benign_fpr_max),
            "gate_adv_malicious_recall_min": float(gate_adv_malicious_recall_min),
            "gate_epsilon": float(gate_epsilon),
            "selected_gate_pass": bool(gate_pass),
            "selected_gate_failure_category": gate_failure_category,
            "selected_gate_flags": {
                "gate_clean_fpr": bool(selected_gate["gate_clean_fpr"]),
                "gate_attacked_benign_fpr": bool(selected_gate["gate_attacked_benign_fpr"]),
                "gate_adv_malicious_recall": bool(selected_gate["gate_adv_malicious_recall"]),
            },
        },
        "files": {},
    }

    if out_family_dir is not None:
        out_family_dir.mkdir(parents=True, exist_ok=True)
        out_models_dir = out_family_dir / "models"
        out_models_dir.mkdir(parents=True, exist_ok=True)

        for name in ["metrics_summary.json", "metrics_summary.csv", "metrics_summary_per_protocol_models.csv"]:
            src = base_run_dir / name
            dst = out_family_dir / name
            if src.exists():
                shutil.copy2(src, dst)

        for model_file in (base_run_dir / "models").glob("*"):
            if model_file.is_file():
                shutil.copy2(model_file, out_models_dir / model_file.name)

        wifi_model_path = out_models_dir / "wifi__xgboost_tuned.json"
        wifi_model.save_model(str(wifi_model_path))

        thresholds_new = dict(thresholds_by_protocol)
        thresholds_new["wifi"] = float(wifi_threshold_new)
        thresholds_path = out_family_dir / "thresholds_by_protocol.json"
        with thresholds_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): float(v) for k, v in sorted(thresholds_new.items())}, f, indent=2)

        hardneg_stats_rows = train_campaign["stats_rows"] + val_campaign["stats_rows"] + [mal_eval["stats_row"]]
        hardneg_stats_df = pd.DataFrame(hardneg_stats_rows)
        hardneg_stats_path = out_family_dir / "hard_negative_stats.csv"
        hardneg_stats_df.to_csv(hardneg_stats_path, index=False)

        preview_rows = train_campaign["preview_rows"] + val_campaign["preview_rows"] + mal_eval["preview_rows"]
        if preview_rows:
            preview_df = pd.DataFrame(preview_rows).head(max(1, int(preview_max_rows)))
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
        preview_path = out_family_dir / "hard_negative_preview.csv"
        preview_df.to_csv(preview_path, index=False)

        constraints_summary_path = out_family_dir / "hardening_constraints_summary.csv"
        constraints_summary_df.to_csv(constraints_summary_path, index=False)
        realism_profile_path = out_family_dir / "hardening_realism_profile.json"
        with realism_profile_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(realism_profile_json), f, indent=2)

        notice_path = out_family_dir / "NON_AUTHORITATIVE_METRICS_NOTICE.txt"
        notice_path.write_text(
            (
                "The copied metrics_summary*.json/csv files are inherited from base run for compatibility only.\n"
                "Use decision_table.csv and family_summary.json in matrix output as authoritative.\n"
            ),
            encoding="utf-8",
        )

        family_summary_path = out_family_dir / "family_summary.json"
        with family_summary_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(out_payload), f, indent=2)

        out_payload["files"] = {
            "family_dir": str(out_family_dir),
            "wifi_model": str(wifi_model_path),
            "thresholds_by_protocol": str(thresholds_path),
            "hard_negative_stats_csv": str(hardneg_stats_path),
            "hard_negative_preview_csv": str(preview_path),
            "hardening_constraints_summary_csv": str(constraints_summary_path),
            "hardening_realism_profile_json": str(realism_profile_path),
            "non_authoritative_notice": str(notice_path),
            "family_summary_json": str(family_summary_path),
        }

    return out_payload

def main() -> None:
    start_ts = time.time()
    args = parse_args()
    epsilons = parse_float_csv(args.hardneg_epsilons)
    gate_epsilon = float(args.gate_epsilon)
    if not any(abs(float(e) - gate_epsilon) <= 1e-12 for e in epsilons):
        epsilons = sorted(set(epsilons + [gate_epsilon]))

    if str(args.stability_extra_seeds).strip():
        stability_seeds = parse_int_csv(str(args.stability_extra_seeds), allow_empty=False)
    else:
        stability_seeds = []

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
    log_progress(f"gate_epsilon={gate_epsilon}", start_ts=start_ts)

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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run_dir = out_root / f"{base_run_dir.name}_wifi_rebalance_matrix_v1_{ts}"
    out_run_dir.mkdir(parents=True, exist_ok=True)
    families_dir = out_run_dir / "families"
    families_dir.mkdir(parents=True, exist_ok=True)

    family_cfgs = _family_matrix_configs(args)
    family_results: List[Dict[str, Any]] = []
    for family_ix, family_cfg in enumerate(family_cfgs):
        family_seed = int(args.seed) + family_ix * 10_000
        family_dir = families_dir / f"{family_cfg['family_id']}_{family_cfg['family_name']}"
        family_result = _run_single_family(
            family_cfg=family_cfg,
            run_seed=family_seed,
            start_ts=start_ts,
            out_family_dir=family_dir,
            base_run_dir=base_run_dir,
            thresholds_by_protocol=thresholds_by_protocol,
            wifi_threshold_base=wifi_threshold_base,
            feature_columns=feature_columns,
            wifi_hparams=wifi_hparams,
            x_train=x_train,
            y_train=y_train,
            src_train_relpath=src_train_relpath,
            src_train_row_index=src_train_row_index,
            x_val=x_val,
            y_val=y_val,
            src_val_relpath=src_val_relpath,
            src_val_row_index=src_val_row_index,
            train_benign_idx=train_benign_idx,
            val_benign_idx=val_benign_idx,
            val_malicious_idx=val_malicious_idx,
            wifi_base_booster=wifi_base_booster,
            wifi_profile=wifi_profile,
            constraints_summary_df=constraints_summary_df,
            realism_profile_json=realism_profile_json,
            epsilons=epsilons,
            gate_epsilon=gate_epsilon,
            query_budget_benign=query_budget_benign,
            query_max_steps_benign=query_max_steps_benign,
            query_candidates_benign=query_candidates_benign,
            query_subset_benign=query_subset_benign,
            query_budget_mal=query_budget_mal,
            query_max_steps_mal=query_max_steps_mal,
            query_candidates_mal=query_candidates_mal,
            query_subset_mal=query_subset_mal,
            threshold_grid_size=int(args.threshold_grid_size),
            gate_clean_fpr_max=float(args.gate_clean_fpr_max),
            gate_attacked_benign_fpr_max=float(args.gate_attacked_benign_fpr_max),
            gate_adv_malicious_recall_min=float(args.gate_adv_malicious_recall_min),
            xgb_device=str(args.xgb_device),
            preview_max_rows=int(args.preview_max_rows),
        )
        family_results.append(family_result)

    decision_rows: List[Dict[str, Any]] = []
    for fr in family_results:
        selected = fr["thresholds"]["selection"]["selected_metrics"]
        robust_new = fr["validation_metrics"]["robust_new_threshold"]
        decision_rows.append(
            {
                "family_id": str(fr["family_id"]),
                "family_name": str(fr["family_name"]),
                "family_description": str(fr["family_description"]),
                "seed": int(fr["seed"]),
                "selected_threshold": float(fr["thresholds"]["wifi_new"]),
                "clean_fpr": float(selected["clean_fpr"]),
                "attacked_benign_fpr": float(selected["attacked_benign_fpr"]),
                "adv_malicious_recall": float(selected["attacked_malicious_recall"]),
                "clean_f1": float(selected["clean_f1"]),
                "robust_f1": float(robust_new["f1"]) if np.isfinite(robust_new["f1"]) else float("nan"),
                "gate_pass": bool(selected["gate_pass"]),
                "gate_failure_category": str(selected["gate_failure_category"]),
                "gate_clean_fpr": bool(selected["gate_clean_fpr"]),
                "gate_attacked_benign_fpr": bool(selected["gate_attacked_benign_fpr"]),
                "gate_adv_malicious_recall": bool(selected["gate_adv_malicious_recall"]),
                "family_summary_json": str(fr["files"].get("family_summary_json", "")),
            }
        )

    ranked_rows = sorted(
        decision_rows,
        key=lambda r: (
            0 if bool(r["gate_pass"]) else 1,
            float(r["attacked_benign_fpr"]) if np.isfinite(r["attacked_benign_fpr"]) else float("inf"),
            float(r["clean_fpr"]) if np.isfinite(r["clean_fpr"]) else float("inf"),
            -(float(r["adv_malicious_recall"]) if np.isfinite(r["adv_malicious_recall"]) else -float("inf")),
            -(float(r["clean_f1"]) if np.isfinite(r["clean_f1"]) else -float("inf")),
            -(float(r["robust_f1"]) if np.isfinite(r["robust_f1"]) else -float("inf")),
        ),
    )
    for rank_ix, row in enumerate(ranked_rows, start=1):
        row["final_rank"] = int(rank_ix)

    winner_row = ranked_rows[0]
    winner_family_id = str(winner_row["family_id"])
    winner_cfg = next(cfg for cfg in family_cfgs if str(cfg["family_id"]) == winner_family_id)

    stability_rows: List[Dict[str, Any]] = [
        {
            "seed": int(winner_row["seed"]),
            "family_id": str(winner_row["family_id"]),
            "family_name": str(winner_row["family_name"]),
            "gate_pass": bool(winner_row["gate_pass"]),
            "clean_fpr": float(winner_row["clean_fpr"]),
            "attacked_benign_fpr": float(winner_row["attacked_benign_fpr"]),
            "adv_malicious_recall": float(winner_row["adv_malicious_recall"]),
            "source": "matrix_primary",
        }
    ]
    if stability_seeds:
        stability_dir = out_run_dir / "stability"
        stability_dir.mkdir(parents=True, exist_ok=True)
        for extra_seed in stability_seeds:
            stab_result = _run_single_family(
                family_cfg=winner_cfg,
                run_seed=int(extra_seed),
                start_ts=start_ts,
                out_family_dir=stability_dir / f"seed_{int(extra_seed)}",
                base_run_dir=base_run_dir,
                thresholds_by_protocol=thresholds_by_protocol,
                wifi_threshold_base=wifi_threshold_base,
                feature_columns=feature_columns,
                wifi_hparams=wifi_hparams,
                x_train=x_train,
                y_train=y_train,
                src_train_relpath=src_train_relpath,
                src_train_row_index=src_train_row_index,
                x_val=x_val,
                y_val=y_val,
                src_val_relpath=src_val_relpath,
                src_val_row_index=src_val_row_index,
                train_benign_idx=train_benign_idx,
                val_benign_idx=val_benign_idx,
                val_malicious_idx=val_malicious_idx,
                wifi_base_booster=wifi_base_booster,
                wifi_profile=wifi_profile,
                constraints_summary_df=constraints_summary_df,
                realism_profile_json=realism_profile_json,
                epsilons=epsilons,
                gate_epsilon=gate_epsilon,
                query_budget_benign=query_budget_benign,
                query_max_steps_benign=query_max_steps_benign,
                query_candidates_benign=query_candidates_benign,
                query_subset_benign=query_subset_benign,
                query_budget_mal=query_budget_mal,
                query_max_steps_mal=query_max_steps_mal,
                query_candidates_mal=query_candidates_mal,
                query_subset_mal=query_subset_mal,
                threshold_grid_size=int(args.threshold_grid_size),
                gate_clean_fpr_max=float(args.gate_clean_fpr_max),
                gate_attacked_benign_fpr_max=float(args.gate_attacked_benign_fpr_max),
                gate_adv_malicious_recall_min=float(args.gate_adv_malicious_recall_min),
                xgb_device=str(args.xgb_device),
                preview_max_rows=int(args.preview_max_rows),
            )
            selected = stab_result["thresholds"]["selection"]["selected_metrics"]
            stability_rows.append(
                {
                    "seed": int(extra_seed),
                    "family_id": str(stab_result["family_id"]),
                    "family_name": str(stab_result["family_name"]),
                    "gate_pass": bool(selected["gate_pass"]),
                    "clean_fpr": float(selected["clean_fpr"]),
                    "attacked_benign_fpr": float(selected["attacked_benign_fpr"]),
                    "adv_malicious_recall": float(selected["attacked_malicious_recall"]),
                    "source": "stability_rerun",
                }
            )

    any_gate_pass = any(bool(r["gate_pass"]) for r in ranked_rows)
    stability_consistent = all(bool(r["gate_pass"]) for r in stability_rows)
    escalate = False
    escalate_reason = "none"
    if not any_gate_pass:
        escalate = True
        escalate_reason = "no_family_passed_gates"
    elif stability_seeds and not stability_consistent:
        escalate = True
        escalate_reason = "winner_not_stable"

    decision_df = pd.DataFrame(ranked_rows)
    decision_csv = out_run_dir / "decision_table.csv"
    decision_df.to_csv(decision_csv, index=False)
    decision_json = out_run_dir / "decision_table.json"
    with decision_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(ranked_rows), f, indent=2)

    stability_df = pd.DataFrame(stability_rows)
    stability_csv = out_run_dir / "stability_check.csv"
    stability_df.to_csv(stability_csv, index=False)
    stability_json = out_run_dir / "stability_check.json"
    with stability_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(stability_rows), f, indent=2)

    escalation = {
        "triggered": bool(escalate),
        "reason": str(escalate_reason),
        "recommendation": (
            "add_benign_external_data_then_rerun_same_gate_profile"
            if bool(escalate)
            else "no_data_expansion_needed_yet"
        ),
    }
    escalation_json = out_run_dir / "escalation_recommendation.json"
    with escalation_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(escalation), f, indent=2)

    top_constraints = out_run_dir / "hardening_constraints_summary.csv"
    constraints_summary_df.to_csv(top_constraints, index=False)
    top_profile = out_run_dir / "hardening_realism_profile.json"
    with top_profile.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(realism_profile_json), f, indent=2)

    notice_path = out_run_dir / "AUTHORITATIVE_RESULTS_NOTICE.txt"
    notice_path.write_text(
        (
            "Authoritative: decision_table.csv, stability_check.csv, escalation_recommendation.json.\n"
            "Family-level copied metrics_summary*.json/csv are compatibility artifacts from base run.\n"
        ),
        encoding="utf-8",
    )

    matrix_summary = {
        "generated_at": datetime.now().isoformat(),
        "base_run_dir": str(base_run_dir),
        "output_run_dir": str(out_run_dir),
        "xgb_booster_inference_device": str(resolved_device),
        "feature_count": int(len(feature_columns)),
        "attack_profile": {
            "mode": "fixed_worst_case_query",
            "gate_epsilon": float(gate_epsilon),
            "query_budget_benign": int(query_budget_benign),
            "query_budget_malicious": int(query_budget_mal),
            "query_max_steps_benign": int(query_max_steps_benign),
            "query_max_steps_malicious": int(query_max_steps_mal),
            "query_candidates_benign": int(query_candidates_benign),
            "query_candidates_malicious": int(query_candidates_mal),
            "query_feature_subset_benign": int(query_subset_benign),
            "query_feature_subset_malicious": int(query_subset_mal),
        },
        "gates": {
            "clean_fpr_max": float(args.gate_clean_fpr_max),
            "attacked_benign_fpr_max": float(args.gate_attacked_benign_fpr_max),
            "adv_malicious_recall_min": float(args.gate_adv_malicious_recall_min),
        },
        "ranking_policy": (
            "gate_pass_then_min_attacked_benign_fpr_then_min_clean_fpr_"
            "then_max_adv_malicious_recall_then_max_clean_f1"
        ),
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
        "families": to_jsonable(family_results),
        "decision_table_rows": to_jsonable(ranked_rows),
        "winner": to_jsonable(winner_row),
        "stability": {
            "extra_seeds": [int(s) for s in stability_seeds],
            "rows": to_jsonable(stability_rows),
            "consistent_gate_pass": bool(stability_consistent),
        },
        "escalation_recommendation": escalation,
        "files": {
            "decision_table_csv": str(decision_csv),
            "decision_table_json": str(decision_json),
            "stability_check_csv": str(stability_csv),
            "stability_check_json": str(stability_json),
            "escalation_recommendation_json": str(escalation_json),
            "hardening_constraints_summary_csv": str(top_constraints),
            "hardening_realism_profile_json": str(top_profile),
            "authoritative_notice": str(notice_path),
        },
    }
    matrix_summary_path = out_run_dir / "matrix_summary.json"
    with matrix_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(matrix_summary), f, indent=2)

    log_progress(f"saved matrix run: {out_run_dir}", start_ts=start_ts)
    log_progress(f"saved decision table: {decision_csv}", start_ts=start_ts)
    log_progress(f"saved matrix summary: {matrix_summary_path}", start_ts=start_ts)
    print(str(out_run_dir))

@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    family_id: str
    family_name: str

    @property
    def key(self) -> str:
        return f"{self.model_name}__{self.family_id}"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_")


def _apply_external_benign_rows(
    *,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    feature_columns: List[str],
    protocols: Sequence[str],
    start_ts: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csvs = (
        [str(Path(p).resolve()) for p in parse_str_csv(args.external_benign_csvs)]
        if str(args.external_benign_csvs).strip()
        else []
    )
    if not csvs:
        return train_df, {
            "enabled": False,
            "external_csvs": [],
            "rows_raw": 0,
            "rows_kept": 0,
            "rows_dropped_non_benign_or_protocol": 0,
            "rows_dropped_overlap": 0,
        }

    usecols = ["source_relpath", "source_row_index", "protocol_hint", "label"] + feature_columns
    pieces: List[pd.DataFrame] = []
    rows_raw = 0
    rows_kept_raw = 0
    rows_drop_basic = 0
    allowed = {protocol_slug(p) for p in protocols}

    for raw in csvs:
        p = Path(raw)
        if not p.exists():
            raise RuntimeError(f"External benign CSV not found: {p}")
        df = pd.read_csv(p, usecols=usecols)
        rows_raw += int(len(df))
        proto = df["protocol_hint"].fillna("").astype(str).map(protocol_slug)
        y = normalize_label_series(df["label"])
        mask = (y == 0) & proto.isin(allowed)
        rows_drop_basic += int(np.sum(~mask.to_numpy()))
        if int(np.sum(mask.to_numpy())) <= 0:
            continue
        sub = df.loc[mask, ["source_relpath", "source_row_index"] + feature_columns].copy()
        sub["protocol_hint"] = proto[mask].to_numpy()
        sub["label"] = np.zeros(int(len(sub)), dtype=np.int8)
        pieces.append(sub)
        rows_kept_raw += int(len(sub))

    if not pieces:
        return train_df, {
            "enabled": True,
            "external_csvs": csvs,
            "rows_raw": int(rows_raw),
            "rows_kept": 0,
            "rows_dropped_non_benign_or_protocol": int(rows_drop_basic),
            "rows_dropped_overlap": 0,
        }

    ext_df = pd.concat(pieces, ignore_index=True)
    if int(args.external_benign_max_rows) > 0 and len(ext_df) > int(args.external_benign_max_rows):
        rng = np.random.default_rng(int(args.seed) + 91573)
        pick = rng.choice(len(ext_df), size=int(args.external_benign_max_rows), replace=False)
        pick.sort()
        ext_df = ext_df.iloc[pick].reset_index(drop=True)

    overlap_drop = 0
    if bool(args.external_benign_overlap_check):
        base_keys = (
            train_df["source_relpath"].astype(str).fillna("")
            + "::"
            + pd.to_numeric(train_df["source_row_index"], errors="coerce").fillna(-1).astype(np.int64).astype(str)
        )
        ext_keys = (
            ext_df["source_relpath"].astype(str).fillna("")
            + "::"
            + pd.to_numeric(ext_df["source_row_index"], errors="coerce").fillna(-1).astype(np.int64).astype(str)
        )
        overlap_mask = ext_keys.isin(set(base_keys.tolist())).to_numpy()
        overlap_drop = int(np.sum(overlap_mask))
        if overlap_drop > 0:
            ext_df = ext_df.loc[~overlap_mask].reset_index(drop=True)

    merged = pd.concat([train_df, ext_df], ignore_index=True)
    log_progress(f"external benign rows appended: {len(ext_df)}", start_ts=start_ts)
    return merged, {
        "enabled": True,
        "external_csvs": csvs,
        "rows_raw": int(rows_raw),
        "rows_kept": int(len(ext_df)),
        "rows_kept_pre_overlap": int(rows_kept_raw),
        "rows_dropped_non_benign_or_protocol": int(rows_drop_basic),
        "rows_dropped_overlap": int(overlap_drop),
    }


def _run_protocol_model_family_candidate(
    *,
    protocol: str,
    model_name: str,
    family_cfg: Dict[str, Any],
    run_seed: int,
    sampling_seed_used: int,
    stability_sampling_mode: str,
    out_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    src_train_relpath: np.ndarray,
    src_train_row_index: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    src_val_relpath: np.ndarray,
    src_val_row_index: np.ndarray,
    train_benign_idx: np.ndarray,
    train_malicious_idx: np.ndarray,
    val_benign_idx: np.ndarray,
    val_malicious_idx: np.ndarray,
    attack_booster: xgb.Booster,
    attack_threshold: float,
    attack_source_mode: str,
    profile: Dict[str, Any],
    model_hparams: Dict[str, Any],
    stage_name: str,
    stage_search_profile: Dict[str, Any],
    epsilons: Sequence[float],
    gate_epsilon: float,
    query_budget_benign: int,
    query_max_steps_benign: int,
    query_candidates_benign: int,
    query_subset_benign: int,
    query_score_batch_benign: int,
    query_fast_projection: bool,
    query_refine_topk: int,
    query_max_active_rows_benign: Optional[int],
    query_stagnation_patience_benign: Optional[int],
    query_stagnation_min_delta_benign: float,
    query_budget_mal: int,
    query_max_steps_mal: int,
    query_candidates_mal: int,
    query_subset_mal: int,
    query_score_batch_mal: int,
    query_max_active_rows_mal: Optional[int],
    query_stagnation_patience_mal: Optional[int],
    query_stagnation_min_delta_mal: float,
    threshold_grid_size: int,
    gate_clean_fpr_max: float,
    gate_attacked_benign_fpr_max: float,
    gate_adv_malicious_recall_min: float,
    threshold_gate_attacked_benign_margin: float,
    strict_fpr_feasibility_check: bool,
    min_val_benign_for_fpr_gate: int,
    split_seed_used: int,
    xgb_device: str,
    mlp_epochs: int,
    mlp_predict_batch_size: int,
    bluetooth_hardneg_max_fraction: float,
    preview_max_rows: int,
    save_models: bool,
    start_ts: float,
) -> Dict[str, Any]:
    proto = protocol_slug(protocol)
    fam_id = str(family_cfg["family_id"])
    fam_name = str(family_cfg["family_name"])
    extra_topk = int(family_cfg["extra_topk_per_epsilon"])
    hardneg_weight = float(family_cfg["hardneg_weight"])
    maladv_weight = float(family_cfg["maladv_weight"])

    out_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"candidate[{proto}:{model_name}:{fam_id}] seed={run_seed}", start_ts=start_ts)
    candidate_t0 = time.time()
    warm_fit_s = 0.0
    attack_gen_s = 0.0
    robust_fit_s = 0.0
    threshold_eval_s = 0.0

    warm_t0 = time.time()
    warm_model = _fit_model_from_scratch(
        protocol=proto,
        model_name=model_name,
        model_hparams=model_hparams,
        x_train=x_train,
        y_train=y_train,
        sample_weight=np.ones(y_train.shape[0], dtype=np.float32),
        x_val=x_val,
        y_val=y_val,
        seed=int(run_seed),
        requested_device=str(xgb_device),
        mlp_epochs=int(mlp_epochs),
        mlp_predict_batch_size=int(mlp_predict_batch_size),
        start_ts=start_ts,
    )
    warm_fit_s = float(time.time() - warm_t0)
    source_mode = str(attack_source_mode).strip().lower()
    if source_mode not in {"fixed_xgb", "candidate_model"}:
        raise RuntimeError(f"Unsupported resolved attack source mode: {attack_source_mode}")

    threshold_t0 = time.time()
    p_val_warm = warm_model.predict_proba_fn(x_val.astype(np.float32, copy=False))
    warm_threshold = _select_base_threshold_from_clean_fpr(
        y_val=y_val,
        p_val=p_val_warm,
        max_fpr=float(max(1e-6, float(gate_clean_fpr_max))),
    )
    threshold_eval_s += float(time.time() - threshold_t0)
    if source_mode == "candidate_model":
        prefit_attack_scorer = _make_model_attack_scorer(
            model_bundle=warm_model,
            threshold=float(attack_threshold),
            source_detail=f"candidate_model:warm_{proto}_{model_name}",
        )
    else:
        prefit_attack_scorer = _make_xgb_attack_scorer(
            booster=attack_booster,
            threshold=float(attack_threshold),
            source_detail=f"fixed_xgb:{proto}_base",
        )

    attack_t0 = time.time()
    train_benign_campaign = _run_benign_query_campaign(
        split_name=f"train_{proto}_{model_name}_{fam_name}",
        x_benign=x_train[train_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_train_relpath[train_benign_idx],
        src_row_index=src_train_row_index[train_benign_idx],
        attack_scorer=prefit_attack_scorer,
        threshold=float(attack_threshold),
        profile=profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        query_score_batch_rows=query_score_batch_benign,
        query_fast_projection=bool(query_fast_projection),
        query_refine_topk=max(1, int(query_refine_topk)),
        query_max_active_rows_per_step=query_max_active_rows_benign,
        query_stagnation_patience=query_stagnation_patience_benign,
        query_stagnation_min_delta=query_stagnation_min_delta_benign,
        extra_topk_per_epsilon=extra_topk,
        seed=int(run_seed) + 10_101,
        start_ts=start_ts,
    )
    train_malicious_campaign = _run_malicious_query_eval(
        split_name=f"train_{proto}_{model_name}_{fam_name}",
        x_malicious=x_train[train_malicious_idx].astype(np.float32, copy=False),
        src_relpath=src_train_relpath[train_malicious_idx],
        src_row_index=src_train_row_index[train_malicious_idx],
        attack_scorer=prefit_attack_scorer,
        threshold=float(attack_threshold),
        profile=profile,
        epsilon=float(gate_epsilon),
        query_budget=query_budget_mal,
        query_max_steps=query_max_steps_mal,
        query_candidates_per_step=query_candidates_mal,
        query_feature_subset_size=query_subset_mal,
        query_score_batch_rows=query_score_batch_mal,
        query_fast_projection=bool(query_fast_projection),
        query_refine_topk=max(1, int(query_refine_topk)),
        query_max_active_rows_per_step=query_max_active_rows_mal,
        query_stagnation_patience=query_stagnation_patience_mal,
        query_stagnation_min_delta=query_stagnation_min_delta_mal,
        seed=int(run_seed) + 20_201,
        start_ts=start_ts,
    )
    attack_gen_s += float(time.time() - attack_t0)

    x_hardneg = train_benign_campaign["selected_adv"].astype(np.float32, copy=False)
    x_maladv = train_malicious_campaign["x_adv"].astype(np.float32, copy=False)
    x_hardneg, hardneg_cap_info = _cap_bluetooth_hardneg_rows(
        protocol=proto,
        x_hardneg=x_hardneg,
        sampled_benign_count=int(train_benign_idx.shape[0]),
        max_fraction=float(bluetooth_hardneg_max_fraction),
        seed=int(run_seed) + 92_003,
    )
    if x_hardneg.shape[0] <= 0 and x_maladv.shape[0] <= 0:
        raise RuntimeError(f"No robust augmentations generated for [{proto}:{model_name}:{fam_id}]")

    blocks = [x_train.astype(np.float32, copy=False)]
    labels = [y_train.astype(np.int8, copy=False)]
    weights = [np.ones(y_train.shape[0], dtype=np.float32)]
    if x_hardneg.shape[0] > 0:
        blocks.append(x_hardneg)
        labels.append(np.zeros(x_hardneg.shape[0], dtype=np.int8))
        weights.append(np.full(x_hardneg.shape[0], hardneg_weight, dtype=np.float32))
    if x_maladv.shape[0] > 0:
        blocks.append(x_maladv)
        labels.append(np.ones(x_maladv.shape[0], dtype=np.int8))
        weights.append(np.full(x_maladv.shape[0], maladv_weight, dtype=np.float32))

    x_aug = np.vstack(blocks).astype(np.float32, copy=False)
    y_aug = np.concatenate(labels, axis=0).astype(np.int8, copy=False)
    w_aug = np.concatenate(weights, axis=0).astype(np.float32, copy=False)

    robust_t0 = time.time()
    robust_model = _fit_model_from_scratch(
        protocol=proto,
        model_name=model_name,
        model_hparams=model_hparams,
        x_train=x_aug,
        y_train=y_aug,
        sample_weight=w_aug,
        x_val=x_val,
        y_val=y_val,
        seed=int(run_seed) + 70_007,
        requested_device=str(xgb_device),
        mlp_epochs=int(mlp_epochs),
        mlp_predict_batch_size=int(mlp_predict_batch_size),
        start_ts=start_ts,
    )
    robust_fit_s = float(time.time() - robust_t0)
    if source_mode == "candidate_model":
        val_attack_scorer = _make_model_attack_scorer(
            model_bundle=robust_model,
            threshold=float(attack_threshold),
            source_detail=f"candidate_model:robust_{proto}_{model_name}",
        )
    else:
        val_attack_scorer = prefit_attack_scorer

    attack_t0 = time.time()
    val_benign_campaign = _run_benign_query_campaign(
        split_name=f"val_{proto}_{model_name}",
        x_benign=x_val[val_benign_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_benign_idx],
        src_row_index=src_val_row_index[val_benign_idx],
        attack_scorer=val_attack_scorer,
        threshold=float(attack_threshold),
        profile=profile,
        epsilons=epsilons,
        query_budget=query_budget_benign,
        query_max_steps=query_max_steps_benign,
        query_candidates_per_step=query_candidates_benign,
        query_feature_subset_size=query_subset_benign,
        query_score_batch_rows=query_score_batch_benign,
        query_fast_projection=bool(query_fast_projection),
        query_refine_topk=max(1, int(query_refine_topk)),
        query_max_active_rows_per_step=query_max_active_rows_benign,
        query_stagnation_patience=query_stagnation_patience_benign,
        query_stagnation_min_delta=query_stagnation_min_delta_benign,
        extra_topk_per_epsilon=0,
        seed=int(run_seed) + 30_303,
        start_ts=start_ts,
    )
    val_malicious_campaign = _run_malicious_query_eval(
        split_name=f"val_{proto}_{model_name}",
        x_malicious=x_val[val_malicious_idx].astype(np.float32, copy=False),
        src_relpath=src_val_relpath[val_malicious_idx],
        src_row_index=src_val_row_index[val_malicious_idx],
        attack_scorer=val_attack_scorer,
        threshold=float(attack_threshold),
        profile=profile,
        epsilon=float(gate_epsilon),
        query_budget=query_budget_mal,
        query_max_steps=query_max_steps_mal,
        query_candidates_per_step=query_candidates_mal,
        query_feature_subset_size=query_subset_mal,
        query_score_batch_rows=query_score_batch_mal,
        query_fast_projection=bool(query_fast_projection),
        query_refine_topk=max(1, int(query_refine_topk)),
        query_max_active_rows_per_step=query_max_active_rows_mal,
        query_stagnation_patience=query_stagnation_patience_mal,
        query_stagnation_min_delta=query_stagnation_min_delta_mal,
        seed=int(run_seed) + 40_404,
        start_ts=start_ts,
    )
    attack_gen_s += float(time.time() - attack_t0)

    threshold_t0 = time.time()
    p_val_clean = robust_model.predict_proba_fn(x_val.astype(np.float32, copy=False))
    gate_eps_mask = np.isclose(
        val_benign_campaign["all_adv_eps"].astype(np.float64, copy=False),
        float(gate_epsilon),
        atol=1e-12,
        rtol=0.0,
    )
    if int(np.sum(gate_eps_mask)) <= 0:
        raise RuntimeError(f"No attacked-benign rows at gate_epsilon for [{proto}:{model_name}:{fam_id}]")
    x_val_adv_benign = val_benign_campaign["all_adv"][gate_eps_mask].astype(np.float32, copy=False)
    p_val_adv_benign = robust_model.predict_proba_fn(x_val_adv_benign)
    p_val_adv_malicious = robust_model.predict_proba_fn(val_malicious_campaign["x_adv"].astype(np.float32, copy=False))

    threshold_sel = _select_threshold_gate_lexicographic(
        y_clean=y_val,
        p_clean=p_val_clean,
        p_adv_benign=p_val_adv_benign,
        p_adv_malicious=p_val_adv_malicious,
        grid_size=int(threshold_grid_size),
        base_threshold=float(warm_threshold),
        gate_clean_fpr_max=float(gate_clean_fpr_max),
        gate_attacked_benign_fpr_max=float(gate_attacked_benign_fpr_max),
        gate_adv_malicious_recall_min=float(gate_adv_malicious_recall_min),
        gate_attacked_benign_margin=float(threshold_gate_attacked_benign_margin),
    )
    selected = threshold_sel["selected_metrics"]
    val_benign_count = int(val_benign_idx.shape[0])
    val_malicious_count = int(val_malicious_idx.shape[0])
    fpr_feasibility = _compute_fpr_feasibility(
        val_benign_count=val_benign_count,
        min_required=int(min_val_benign_for_fpr_gate),
        enabled=bool(strict_fpr_feasibility_check),
    )
    selected = _apply_data_cap_gate_override(
        selected_metrics=selected,
        strict_fpr_feasible=bool(fpr_feasibility["strict_fpr_feasible"]),
    )
    thr_old = float(warm_threshold)
    thr_new = float(threshold_sel["selected_threshold"])
    robust_new = _combine_robust_metrics(
        p_adv_benign=p_val_adv_benign,
        p_adv_malicious=p_val_adv_malicious,
        threshold=thr_new,
    )
    threshold_eval_s += float(time.time() - threshold_t0)

    stats_rows = (
        train_benign_campaign["stats_rows"]
        + [train_malicious_campaign["stats_row"]]
        + val_benign_campaign["stats_rows"]
        + [val_malicious_campaign["stats_row"]]
    )
    stats_path = out_dir / "attack_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    preview_rows = (
        train_benign_campaign["preview_rows"]
        + train_malicious_campaign["preview_rows"]
        + val_benign_campaign["preview_rows"]
        + val_malicious_campaign["preview_rows"]
    )
    preview_path = out_dir / "attack_preview.csv"
    (pd.DataFrame(preview_rows).head(max(1, int(preview_max_rows))) if preview_rows else pd.DataFrame()).to_csv(
        preview_path, index=False
    )

    model_path = ""
    if bool(save_models):
        model_path = _save_model_artifact(
            robust_model,
            out_dir / "models" / f"{proto}__{_safe_name(model_name)}__{fam_id}",
        )

    total_elapsed_s = float(time.time() - candidate_t0)

    payload: Dict[str, Any] = {
        "protocol": proto,
        "model_name": str(model_name),
        "family_id": fam_id,
        "family_name": fam_name,
        "family_description": str(family_cfg["family_description"]),
        "seed": int(run_seed),
        "sampling_seed_used": int(sampling_seed_used),
        "stability_sampling_mode": str(stability_sampling_mode),
        "stage_name": str(stage_name),
        "num_gate_pass": int(threshold_sel["num_gate_pass"]),
        "gate_pass": bool(selected["gate_pass"]),
        "gate_failure_category": str(selected["gate_failure_category"]),
        "gate_clean_fpr": bool(selected["gate_clean_fpr"]),
        "gate_attacked_benign_fpr": bool(selected["gate_attacked_benign_fpr"]),
        "gate_adv_malicious_recall": bool(selected["gate_adv_malicious_recall"]),
        "clean_fpr": float(selected["clean_fpr"]),
        "attacked_benign_fpr": float(selected["attacked_benign_fpr"]),
        "adv_malicious_recall": float(selected["attacked_malicious_recall"]),
        "clean_f1": float(selected["clean_f1"]),
        "robust_f1": float(robust_new["f1"]) if np.isfinite(robust_new["f1"]) else float("nan"),
        "warm_threshold": float(thr_old),
        "selected_threshold": float(thr_new),
        "threshold_selection_mode": str(threshold_sel.get("selection_mode", "gate_pass_lexicographic")),
        "threshold_selected_adv_shortfall": float(threshold_sel.get("selected_adv_shortfall", float("nan"))),
        "val_benign_count": int(val_benign_count),
        "val_malicious_count": int(val_malicious_count),
        "fpr_resolution": float(fpr_feasibility["fpr_resolution"]),
        "strict_fpr_feasible": bool(fpr_feasibility["strict_fpr_feasible"]),
        "split_seed_used": int(split_seed_used),
        "warm_model_device": str(warm_model.trained_device),
        "robust_model_device": str(robust_model.trained_device),
        "attack_source_mode": str(source_mode),
        "attack_source_detail_prefit": str(prefit_attack_scorer.source_detail),
        "attack_source_detail_val": str(val_attack_scorer.source_detail),
        "hardneg_weight": float(hardneg_weight),
        "maladv_weight": float(maladv_weight),
        "extra_topk_per_epsilon": int(extra_topk),
        "search_profile": to_jsonable(stage_search_profile),
        "search_train_benign_sample": int(stage_search_profile.get("hardneg_train_benign_sample", 0)),
        "search_train_malicious_sample": int(stage_search_profile.get("hardneg_train_malicious_sample", 0)),
        "search_val_benign_sample": int(stage_search_profile.get("hardneg_val_benign_sample", 0)),
        "search_val_malicious_sample": int(stage_search_profile.get("val_malicious_sample", 0)),
        "search_query_budget_benign": int(stage_search_profile.get("query_budget_benign", 0)),
        "search_query_budget_malicious": int(stage_search_profile.get("query_budget_mal", 0)),
        "search_query_steps_benign": int(stage_search_profile.get("query_max_steps_benign", 0)),
        "search_query_steps_malicious": int(stage_search_profile.get("query_max_steps_mal", 0)),
        "search_query_candidates_benign": int(stage_search_profile.get("query_candidates_benign", 0)),
        "search_query_candidates_malicious": int(stage_search_profile.get("query_candidates_mal", 0)),
        "search_query_score_batch_benign": int(stage_search_profile.get("query_score_batch_benign", 0)),
        "search_query_score_batch_malicious": int(stage_search_profile.get("query_score_batch_mal", 0)),
        "threshold_gate_attacked_benign_margin": float(threshold_gate_attacked_benign_margin),
        "n_aug_benign": int(x_hardneg.shape[0]),
        "n_aug_malicious": int(x_maladv.shape[0]),
        "hardneg_cap_info": hardneg_cap_info,
        "timing_seconds": {
            "total": float(total_elapsed_s),
            "warm_fit": float(warm_fit_s),
            "attack_gen": float(attack_gen_s),
            "robust_fit": float(robust_fit_s),
            "threshold_eval": float(threshold_eval_s),
        },
        "files": {
            "candidate_dir": str(out_dir),
            "attack_stats_csv": str(stats_path),
            "attack_preview_csv": str(preview_path),
            "saved_model": str(model_path),
        },
    }
    with (out_dir / "candidate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    log_progress(
        (
            f"candidate_done[{proto}:{model_name}:{fam_id}] seed={run_seed} "
            f"gate_pass={int(bool(payload['gate_pass']))} "
            f"elapsed={total_elapsed_s:.1f}s warm_fit={warm_fit_s:.1f}s "
            f"attack_gen={attack_gen_s:.1f}s robust_fit={robust_fit_s:.1f}s "
            f"threshold_eval={threshold_eval_s:.1f}s"
        ),
        start_ts=start_ts,
    )
    return payload


def _rank_protocol_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if bool(r["gate_pass"]) else 1,
            float(r["attacked_benign_fpr"]) if np.isfinite(r["attacked_benign_fpr"]) else float("inf"),
            float(r["clean_fpr"]) if np.isfinite(r["clean_fpr"]) else float("inf"),
            -(float(r["adv_malicious_recall"]) if np.isfinite(r["adv_malicious_recall"]) else -float("inf")),
            -(float(r["clean_f1"]) if np.isfinite(r["clean_f1"]) else -float("inf")),
            -(float(r["robust_f1"]) if np.isfinite(r["robust_f1"]) else -float("inf")),
        ),
    )
    for i, row in enumerate(ranked, start=1):
        row["final_rank"] = int(i)
    return ranked


def _build_global_rows(protocol_rows: List[Dict[str, Any]], protocols: Sequence[str]) -> List[Dict[str, Any]]:
    proto_set = {protocol_slug(p) for p in protocols}
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in protocol_rows:
        key = f"{row['model_name']}__{row['family_id']}__{int(row['seed'])}"
        grouped.setdefault(key, {})[str(row["protocol"])] = row

    global_rows: List[Dict[str, Any]] = []
    for key, by_proto in grouped.items():
        model_name, family_id, seed_s = key.split("__")
        has_all = proto_set.issubset(set(by_proto.keys()))
        if has_all:
            rows = [by_proto[p] for p in sorted(proto_set)]
            gate_pass = all(bool(r["gate_pass"]) for r in rows)
            failure = (
                "none"
                if gate_pass
                else ";".join(sorted({str(r["gate_failure_category"]) for r in rows if str(r["gate_failure_category"]) != "none"}))
            )
            worst_attacked = max(float(r["attacked_benign_fpr"]) for r in rows)
            worst_clean = max(float(r["clean_fpr"]) for r in rows)
            worst_adv = min(float(r["adv_malicious_recall"]) for r in rows)
            mean_f1 = float(np.mean([float(r["clean_f1"]) for r in rows]))
            robust_vals = [float(r["robust_f1"]) for r in rows if np.isfinite(float(r["robust_f1"]))]
            mean_robust_f1 = float(np.mean(robust_vals)) if robust_vals else float("nan")
        else:
            gate_pass = False
            failure = "missing_protocol_rows"
            worst_attacked = float("inf")
            worst_clean = float("inf")
            worst_adv = -float("inf")
            mean_f1 = float("nan")
            mean_robust_f1 = float("nan")

        global_rows.append(
            {
                "candidate_key": key,
                "candidate_group_key": f"{model_name}__{family_id}",
                "model_name": model_name,
                "family_id": family_id,
                "seed": int(seed_s),
                "protocols_present": ",".join(sorted(by_proto.keys())),
                "has_all_protocols": bool(has_all),
                "gate_pass": bool(gate_pass),
                "gate_failure_category": str(failure),
                "worst_attacked_benign_fpr": float(worst_attacked),
                "worst_clean_fpr": float(worst_clean),
                "worst_adv_malicious_recall": float(worst_adv),
                "mean_clean_f1": float(mean_f1),
                "mean_robust_f1": float(mean_robust_f1),
            }
        )

    ranked = sorted(
        global_rows,
        key=lambda r: (
            0 if bool(r["gate_pass"]) else 1,
            float(r["worst_attacked_benign_fpr"]) if np.isfinite(r["worst_attacked_benign_fpr"]) else float("inf"),
            float(r["worst_clean_fpr"]) if np.isfinite(r["worst_clean_fpr"]) else float("inf"),
            -(float(r["worst_adv_malicious_recall"]) if np.isfinite(r["worst_adv_malicious_recall"]) else -float("inf")),
            -(float(r["mean_clean_f1"]) if np.isfinite(r["mean_clean_f1"]) else -float("inf")),
            -(float(r["mean_robust_f1"]) if np.isfinite(r["mean_robust_f1"]) else -float("inf")),
        ),
    )
    for i, row in enumerate(ranked, start=1):
        row["final_rank"] = int(i)
    return ranked


def _build_stability_shortlist_specs(
    *,
    global_rows: Sequence[Dict[str, Any]],
    family_cfgs: Sequence[Dict[str, Any]],
    stage2_topk_global: int,
    enabled_models: Sequence[str],
) -> List[CandidateSpec]:
    enabled_order = _validate_matrix_model_list(enabled_models, context="stability enabled models")
    enabled_set = {str(m).strip().lower() for m in enabled_order}
    target_k = max(1, int(stage2_topk_global))
    target_k = max(target_k, len(enabled_order))
    fam_name_by_id = {str(f["family_id"]): str(f["family_name"]) for f in family_cfgs}

    selected_rows: List[Dict[str, Any]] = []
    selected_group_keys: set = set()

    for model_name in enabled_order:
        for row in global_rows:
            if str(row.get("model_name", "")).strip().lower() != str(model_name):
                continue
            group_key = str(row.get("candidate_group_key", f"{row.get('model_name')}__{row.get('family_id')}"))
            if group_key in selected_group_keys:
                continue
            selected_rows.append(dict(row))
            selected_group_keys.add(group_key)
            break

    for row in global_rows:
        if len(selected_rows) >= target_k:
            break
        model_name = str(row.get("model_name", "")).strip().lower()
        if model_name not in enabled_set:
            continue
        group_key = str(row.get("candidate_group_key", f"{row.get('model_name')}__{row.get('family_id')}"))
        if group_key in selected_group_keys:
            continue
        selected_rows.append(dict(row))
        selected_group_keys.add(group_key)

    out: List[CandidateSpec] = []
    for row in selected_rows:
        model_name = str(row.get("model_name", "")).strip().lower()
        family_id = str(row.get("family_id", ""))
        out.append(
            CandidateSpec(
                model_name=model_name,
                family_id=family_id,
                family_name=fam_name_by_id.get(family_id, "unknown"),
            )
        )
    return out


def main_protocol_multimodel() -> None:
    start_ts = time.time()
    args = parse_args()
    np.random.seed(int(args.seed))

    protocols = [protocol_slug(p) for p in parse_str_csv(args.protocols)]
    base_models = _validate_matrix_model_list(
        [str(m).strip().lower() for m in parse_str_csv(args.models)],
        context="--models",
    )
    epsilons = parse_float_csv(args.hardneg_epsilons)
    gate_epsilon = float(args.gate_epsilon)
    if not any(abs(float(e) - gate_epsilon) <= 1e-12 for e in epsilons):
        epsilons = sorted(set(epsilons + [gate_epsilon]))
    stability_seeds = parse_int_csv(str(args.stability_seeds), allow_empty=False) if str(args.stability_seeds).strip() else []
    stability_split_mode = str(args.stability_split_mode).strip().lower()
    if stability_split_mode not in {"shared_val_seed", "per_seed"}:
        raise RuntimeError(f"Unsupported --stability-split-mode: {args.stability_split_mode}")
    stability_sampling_mode = str(args.stability_sampling_mode).strip().lower()
    if stability_sampling_mode not in {"fixed", "per_seed", "hybrid"}:
        raise RuntimeError(f"Unsupported --stability-sampling-mode: {args.stability_sampling_mode}")
    requested_attack_source_mode = str(args.attack_source_mode).strip().lower()
    if requested_attack_source_mode not in {"fixed_xgb", "candidate_model", "hybrid"}:
        raise RuntimeError(f"Unsupported --attack-source-mode: {args.attack_source_mode}")
    shared_split_seed = (
        int(args.stability_val_split_seed)
        if args.stability_val_split_seed is not None
        else int(args.coarse_seed)
    )
    shared_sampling_seed = (
        int(args.stability_sampling_seed)
        if args.stability_sampling_seed is not None
        else int(args.coarse_seed)
    )
    auto_min_val_benign_for_fpr_gate = int(math.ceil(1.0 / max(1e-12, float(args.gate_clean_fpr_max))))
    min_val_benign_for_fpr_gate = (
        int(args.min_val_benign_for_fpr_gate)
        if args.min_val_benign_for_fpr_gate is not None
        else int(auto_min_val_benign_for_fpr_gate)
    )
    if int(min_val_benign_for_fpr_gate) <= 0:
        raise RuntimeError("--min-val-benign-for-fpr-gate must be >= 1")
    strict_fpr_feasibility_check = bool(args.strict_fpr_feasibility_check)
    bluetooth_benign_min_train_rows = int(max(0, int(args.bluetooth_benign_min_train_rows)))
    bluetooth_hardneg_max_fraction = float(args.bluetooth_hardneg_max_fraction)
    if (not math.isfinite(bluetooth_hardneg_max_fraction)) or bluetooth_hardneg_max_fraction < 0.0 or bluetooth_hardneg_max_fraction > 1.0:
        raise RuntimeError("--bluetooth-hardneg-max-fraction must be in [0.0, 1.0]")

    base_run_dir = Path(args.base_run_dir).resolve() if args.base_run_dir else discover_latest_hpo_run(Path("reports"))
    if base_run_dir is None:
        raise RuntimeError("Could not resolve base run directory.")
    base_run_dir = Path(base_run_dir).resolve()
    train_csv = Path(args.train_csv).resolve()
    _ = Path(args.test_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(base_run_dir)
    xgb_boosters = load_xgb_models_by_protocol(base_run_dir)
    _ = configure_boosters_device(
        models_by_protocol=xgb_boosters,
        feature_count=len(feature_columns),
        requested_device=args.xgb_device,
        start_ts=start_ts,
    )
    thresholds_by_protocol = load_thresholds_by_protocol(base_run_dir)
    best_hparams = _load_best_hparams(base_run_dir)
    family_cfgs = _family_matrix_configs(args)

    log_progress(f"base_run_dir={base_run_dir}", start_ts=start_ts)
    log_progress(f"protocols={protocols} base_models={base_models}", start_ts=start_ts)
    log_progress(f"hardneg_epsilons={epsilons} gate_epsilon={gate_epsilon}", start_ts=start_ts)
    log_progress(
        f"attack_source_mode={requested_attack_source_mode} family_pack={args.family_pack} "
        f"bluetooth_hardneg_max_fraction={bluetooth_hardneg_max_fraction}",
        start_ts=start_ts,
    )
    log_progress(
        (
            f"stability_split_mode={stability_split_mode} shared_split_seed={shared_split_seed} "
            f"stability_sampling_mode={stability_sampling_mode} shared_sampling_seed={shared_sampling_seed}"
        ),
        start_ts=start_ts,
    )
    log_progress(
        f"bluetooth_benign_min_train_rows={bluetooth_benign_min_train_rows}",
        start_ts=start_ts,
    )
    log_progress(
        (
            "strict_fpr_feasibility_check="
            f"{int(strict_fpr_feasibility_check)} min_val_benign_for_fpr_gate={int(min_val_benign_for_fpr_gate)} "
            f"(auto_default={int(auto_min_val_benign_for_fpr_gate)})"
        ),
        start_ts=start_ts,
    )
    log_progress(
        f"threshold_gate_attacked_benign_margin={float(args.threshold_gate_attacked_benign_margin):.4f}",
        start_ts=start_ts,
    )

    train_df = _load_protocol_train_rows(
        train_csv=train_csv,
        feature_columns=feature_columns,
        protocols=protocols,
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )
    train_df, external_info = _apply_external_benign_rows(
        args=args,
        train_df=train_df,
        feature_columns=feature_columns,
        protocols=protocols,
        start_ts=start_ts,
    )

    split_bundle_cache: Dict[int, Dict[str, Any]] = {}
    all_fallback_events: List[Dict[str, Any]] = []
    all_data_cap_rows: List[Dict[str, Any]] = []

    def _get_split_bundle(split_seed: int) -> Dict[str, Any]:
        split_seed = int(split_seed)
        if split_seed in split_bundle_cache:
            return split_bundle_cache[split_seed]
        bundle = _prepare_protocol_split_bundle(
            train_df=train_df,
            feature_columns=feature_columns,
            protocols=protocols,
            split_seed=int(split_seed),
            val_mod=int(args.val_mod),
            bluetooth_benign_min_train_rows=int(bluetooth_benign_min_train_rows),
            protocol_max_train_rows=int(args.protocol_max_train_rows),
            protocol_max_val_rows=int(args.protocol_max_val_rows),
            start_ts=start_ts,
        )
        split_bundle_cache[split_seed] = bundle
        all_fallback_events.extend(list(bundle.get("fallback_events", [])))
        all_data_cap_rows.extend(list(bundle.get("data_cap_rows", [])))
        return bundle

    base_split_bundle = _get_split_bundle(shared_split_seed)
    proto_data: Dict[str, Dict[str, Any]] = base_split_bundle["proto_data"]
    fallback_events: List[Dict[str, Any]] = list(base_split_bundle.get("fallback_events", []))
    rng = np.random.default_rng(int(args.seed))

    constraint_blocks: List[pd.DataFrame] = []
    for proto in protocols:
        x_tr = proto_data[proto]["x_train"]
        y_tr = proto_data[proto]["y_train"]
        if int(args.constraint_sample_size) > 0 and x_tr.shape[0] > int(args.constraint_sample_size):
            pick = rng.choice(x_tr.shape[0], size=int(args.constraint_sample_size), replace=False)
            pick.sort()
        else:
            pick = np.arange(x_tr.shape[0], dtype=np.int64)
        df = pd.DataFrame(x_tr[pick], columns=feature_columns)
        df.insert(0, "label", y_tr[pick].astype(np.int8))
        df.insert(0, "protocol_hint", proto)
        constraint_blocks.append(df)
    constraint_sample_df = pd.concat(constraint_blocks, ignore_index=True)

    constraints, constraints_summary_df = build_constraints(
        train_sample_df=constraint_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        percentile_lower=float(args.percentile_lower),
        percentile_upper=float(args.percentile_upper),
    )
    realism_profiles, realism_profile_json = build_realism_profiles(
        train_sample_df=constraint_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        constraints=constraints,
        relation_q_low=float(args.relation_quantile_low),
        relation_q_high=float(args.relation_quantile_high),
        relative_cap_default=float(args.relative_cap_default),
        relative_cap_rate_group=args.relative_cap_rate_group,
        relative_cap_size_group=args.relative_cap_size_group,
        relative_cap_time_group=args.relative_cap_time_group,
    )

    coarse_models = _resolve_stage_models(args, stage_name="coarse")
    coarse_profile = _resolve_stage_search_profile(args, stage_name="coarse")
    stability_profile = _resolve_stage_search_profile(args, stage_name="stability")
    models = list(coarse_models)
    log_progress(f"coarse_models={coarse_models}", start_ts=start_ts)
    log_progress(
        (
            "coarse_profile="
            f"policy={coarse_profile['sampling_policy']} "
            f"samples(train_b={coarse_profile['hardneg_train_benign_sample']},"
            f"train_m={coarse_profile['hardneg_train_malicious_sample']},"
            f"val_b={coarse_profile['hardneg_val_benign_sample']},"
            f"val_m={coarse_profile['val_malicious_sample']}) "
            f"query_b(budget={coarse_profile['query_budget_benign']},"
            f"steps={coarse_profile['query_max_steps_benign']},"
            f"cands={coarse_profile['query_candidates_benign']},"
            f"subset={coarse_profile['query_subset_benign']},"
            f"score_batch_rows={coarse_profile['query_score_batch_benign']}) "
            f"query_m(budget={coarse_profile['query_budget_mal']},"
            f"steps={coarse_profile['query_max_steps_mal']},"
            f"cands={coarse_profile['query_candidates_mal']},"
            f"subset={coarse_profile['query_subset_mal']},"
            f"score_batch_rows={coarse_profile['query_score_batch_mal']}) "
            f"proj(fast={int(bool(coarse_profile.get('query_fast_projection', True)))},"
            f"refine_topk={int(coarse_profile.get('query_refine_topk', 2))})"
        ),
        start_ts=start_ts,
    )
    log_progress(
        (
            "stability_profile="
            f"policy={stability_profile['sampling_policy']} "
            f"samples(train_b={stability_profile['hardneg_train_benign_sample']},"
            f"train_m={stability_profile['hardneg_train_malicious_sample']},"
            f"val_b={stability_profile['hardneg_val_benign_sample']},"
            f"val_m={stability_profile['val_malicious_sample']}) "
            f"query_b(budget={stability_profile['query_budget_benign']},"
            f"steps={stability_profile['query_max_steps_benign']},"
            f"cands={stability_profile['query_candidates_benign']},"
            f"subset={stability_profile['query_subset_benign']},"
            f"score_batch_rows={stability_profile['query_score_batch_benign']}) "
            f"query_m(budget={stability_profile['query_budget_mal']},"
            f"steps={stability_profile['query_max_steps_mal']},"
            f"cands={stability_profile['query_candidates_mal']},"
            f"subset={stability_profile['query_subset_mal']},"
            f"score_batch_rows={stability_profile['query_score_batch_mal']}) "
            f"proj(fast={int(bool(stability_profile.get('query_fast_projection', True)))},"
            f"refine_topk={int(stability_profile.get('query_refine_topk', 2))})"
        ),
        start_ts=start_ts,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run_dir = out_root / f"{base_run_dir.name}_protocol_multimodel_robust_matrix_v1_{ts}"
    out_run_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = out_run_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    protocol_rows: List[Dict[str, Any]] = []
    if args.stage_mode in {"coarse", "both"}:
        coarse_attack_source_mode = _resolve_attack_source_mode_for_stage(requested_attack_source_mode, "coarse")
        for proto in protocols:
            if proto not in xgb_boosters:
                raise RuntimeError(f"Missing base xgboost booster for protocol={proto}")
            attack_booster = xgb_boosters[proto]
            attack_threshold = float(thresholds_by_protocol.get(proto, 0.5))
            pdata = proto_data[proto]
            profile = realism_profiles[proto]
            for model_name in models:
                hparams = dict(best_hparams.get(proto, {}).get(model_name, {}))
                for fam in family_cfgs:
                    seed = int(args.coarse_seed)
                    candidate_dir = candidates_root / f"coarse_{proto}_{_safe_name(model_name)}_{fam['family_id']}_seed_{seed}"
                    sampled_idx = _sample_protocol_indices_for_run(
                        proto=proto,
                        proto_data=pdata,
                        run_seed=int(seed),
                        hardneg_train_benign_sample=int(coarse_profile["hardneg_train_benign_sample"]),
                        hardneg_train_malicious_sample=int(coarse_profile["hardneg_train_malicious_sample"]),
                        hardneg_val_benign_sample=int(coarse_profile["hardneg_val_benign_sample"]),
                        val_malicious_sample=int(coarse_profile["val_malicious_sample"]),
                        sampling_seed=int(seed),
                        sampling_policy=str(coarse_profile.get("sampling_policy", "fixed_count")),
                        sampling_buckets=coarse_profile.get("sampling_buckets"),
                    )
                    row = _run_protocol_model_family_candidate(
                        protocol=proto,
                        model_name=model_name,
                        family_cfg=fam,
                        run_seed=seed,
                        sampling_seed_used=int(seed),
                        stability_sampling_mode="coarse",
                        out_dir=candidate_dir,
                        x_train=pdata["x_train"],
                        y_train=pdata["y_train"],
                        src_train_relpath=pdata["src_train_relpath"],
                        src_train_row_index=pdata["src_train_row_index"],
                        x_val=pdata["x_val"],
                        y_val=pdata["y_val"],
                        src_val_relpath=pdata["src_val_relpath"],
                        src_val_row_index=pdata["src_val_row_index"],
                        train_benign_idx=sampled_idx["train_benign_idx"],
                        train_malicious_idx=sampled_idx["train_malicious_idx"],
                        val_benign_idx=sampled_idx["val_benign_idx"],
                        val_malicious_idx=sampled_idx["val_malicious_idx"],
                        attack_booster=attack_booster,
                        attack_threshold=attack_threshold,
                        attack_source_mode=coarse_attack_source_mode,
                        profile=profile,
                        model_hparams=hparams,
                        stage_name="coarse",
                        stage_search_profile=coarse_profile,
                        epsilons=epsilons,
                        gate_epsilon=gate_epsilon,
                        query_budget_benign=int(coarse_profile["query_budget_benign"]),
                        query_max_steps_benign=int(coarse_profile["query_max_steps_benign"]),
                        query_candidates_benign=int(coarse_profile["query_candidates_benign"]),
                        query_subset_benign=int(coarse_profile["query_subset_benign"]),
                        query_score_batch_benign=int(coarse_profile["query_score_batch_benign"]),
                        query_fast_projection=bool(coarse_profile.get("query_fast_projection", True)),
                        query_refine_topk=max(1, int(coarse_profile.get("query_refine_topk", 2))),
                        query_max_active_rows_benign=coarse_profile["query_max_active_rows_benign"],
                        query_stagnation_patience_benign=coarse_profile["query_stagnation_patience_benign"],
                        query_stagnation_min_delta_benign=float(coarse_profile["query_stagnation_min_delta_benign"]),
                        query_budget_mal=int(coarse_profile["query_budget_mal"]),
                        query_max_steps_mal=int(coarse_profile["query_max_steps_mal"]),
                        query_candidates_mal=int(coarse_profile["query_candidates_mal"]),
                        query_subset_mal=int(coarse_profile["query_subset_mal"]),
                        query_score_batch_mal=int(coarse_profile["query_score_batch_mal"]),
                        query_max_active_rows_mal=coarse_profile["query_max_active_rows_mal"],
                        query_stagnation_patience_mal=coarse_profile["query_stagnation_patience_mal"],
                        query_stagnation_min_delta_mal=float(coarse_profile["query_stagnation_min_delta_mal"]),
                        threshold_grid_size=int(args.threshold_grid_size),
                        gate_clean_fpr_max=float(args.gate_clean_fpr_max),
                        gate_attacked_benign_fpr_max=float(args.gate_attacked_benign_fpr_max),
                        gate_adv_malicious_recall_min=float(args.gate_adv_malicious_recall_min),
                        threshold_gate_attacked_benign_margin=float(args.threshold_gate_attacked_benign_margin),
                        strict_fpr_feasibility_check=bool(strict_fpr_feasibility_check),
                        min_val_benign_for_fpr_gate=int(min_val_benign_for_fpr_gate),
                        split_seed_used=int(pdata.get("split_seed_used", shared_split_seed)),
                        xgb_device=str(args.xgb_device),
                        mlp_epochs=int(args.mlp_epochs),
                        mlp_predict_batch_size=int(args.mlp_predict_batch_size),
                        bluetooth_hardneg_max_fraction=float(bluetooth_hardneg_max_fraction),
                        preview_max_rows=int(args.preview_max_rows),
                        save_models=bool(args.save_models),
                        start_ts=start_ts,
                    )
                    protocol_rows.append(row)

    if args.stage_mode == "stability":
        if not args.coarse_run_dir:
            raise RuntimeError("--coarse-run-dir is required when --stage-mode stability")
        coarse_dir = Path(args.coarse_run_dir).resolve()
        protocol_rows = []
        for proto in protocols:
            p = coarse_dir / f"decision_table_protocol_{proto}.csv"
            if not p.exists():
                raise RuntimeError(f"Missing coarse protocol decision table: {p}")
            protocol_rows.extend(pd.read_csv(p).to_dict(orient="records"))

    per_proto_tables: Dict[str, List[Dict[str, Any]]] = {}
    for proto in protocols:
        rows = [dict(r) for r in protocol_rows if protocol_slug(str(r.get("protocol", ""))) == proto]
        ranked = _rank_protocol_rows(rows)
        per_proto_tables[proto] = ranked
        pd.DataFrame(ranked).to_csv(out_run_dir / f"decision_table_protocol_{proto}.csv", index=False)

    global_rows = _build_global_rows(protocol_rows=protocol_rows, protocols=protocols)
    global_csv = out_run_dir / "decision_table_global.csv"
    pd.DataFrame(global_rows).to_csv(global_csv, index=False)

    shortlist_model_order = [str(r.get("model_name", "")).strip().lower() for r in global_rows]
    stability_models = _resolve_stage_models(
        args,
        stage_name="stability",
        shortlist_models=shortlist_model_order,
    )
    shortlist_specs = _build_stability_shortlist_specs(
        global_rows=global_rows,
        family_cfgs=family_cfgs,
        stage2_topk_global=int(args.stage2_topk_global),
        enabled_models=stability_models,
    )
    if args.stage_mode in {"both", "stability"} and stability_seeds:
        if not shortlist_specs:
            raise RuntimeError(
                "No stability candidates after model filtering. "
                "Check --stability-models and --stage2-topk-global."
            )
        log_progress(
            f"stability_models={sorted({str(spec.model_name).strip().lower() for spec in shortlist_specs})}",
            start_ts=start_ts,
        )

    stability_rows: List[Dict[str, Any]] = []
    if args.stage_mode in {"both", "stability"} and stability_seeds:
        stability_attack_source_mode = _resolve_attack_source_mode_for_stage(requested_attack_source_mode, "stability")
        stability_root = out_run_dir / "stability"
        stability_root.mkdir(parents=True, exist_ok=True)
        fam_by_id = {str(f["family_id"]): f for f in family_cfgs}
        for seed in stability_seeds:
            stability_split_seed = _resolve_stability_split_seed(
                split_mode=stability_split_mode,
                shared_split_seed=int(shared_split_seed),
                stability_seed=int(seed),
            )
            split_bundle = _get_split_bundle(int(stability_split_seed))
            stability_proto_data = split_bundle["proto_data"]
            seed_proto_rows: List[Dict[str, Any]] = []
            for proto in protocols:
                if proto not in xgb_boosters:
                    continue
                attack_booster = xgb_boosters[proto]
                attack_threshold = float(thresholds_by_protocol.get(proto, 0.5))
                pdata = stability_proto_data[proto]
                profile = realism_profiles[proto]
                stability_sampling_seed_for_proto = _resolve_stability_sampling_seed(
                    sampling_mode=stability_sampling_mode,
                    shared_sampling_seed=int(shared_sampling_seed),
                    stability_seed=int(seed),
                    protocol=proto,
                )
                for spec in shortlist_specs:
                    fam = fam_by_id[spec.family_id]
                    hparams = dict(best_hparams.get(proto, {}).get(spec.model_name, {}))
                    candidate_dir = stability_root / f"seed_{seed}_{proto}_{_safe_name(spec.model_name)}_{spec.family_id}"
                    sampled_idx = _sample_protocol_indices_for_run(
                        proto=proto,
                        proto_data=pdata,
                        run_seed=int(seed),
                        hardneg_train_benign_sample=int(stability_profile["hardneg_train_benign_sample"]),
                        hardneg_train_malicious_sample=int(stability_profile["hardneg_train_malicious_sample"]),
                        hardneg_val_benign_sample=int(stability_profile["hardneg_val_benign_sample"]),
                        val_malicious_sample=int(stability_profile["val_malicious_sample"]),
                        sampling_seed=int(stability_sampling_seed_for_proto),
                        sampling_policy=str(stability_profile.get("sampling_policy", "fixed_count")),
                        sampling_buckets=stability_profile.get("sampling_buckets"),
                    )
                    row = _run_protocol_model_family_candidate(
                        protocol=proto,
                        model_name=spec.model_name,
                        family_cfg=fam,
                        run_seed=int(seed),
                        sampling_seed_used=int(stability_sampling_seed_for_proto),
                        stability_sampling_mode=str(stability_sampling_mode),
                        out_dir=candidate_dir,
                        x_train=pdata["x_train"],
                        y_train=pdata["y_train"],
                        src_train_relpath=pdata["src_train_relpath"],
                        src_train_row_index=pdata["src_train_row_index"],
                        x_val=pdata["x_val"],
                        y_val=pdata["y_val"],
                        src_val_relpath=pdata["src_val_relpath"],
                        src_val_row_index=pdata["src_val_row_index"],
                        train_benign_idx=sampled_idx["train_benign_idx"],
                        train_malicious_idx=sampled_idx["train_malicious_idx"],
                        val_benign_idx=sampled_idx["val_benign_idx"],
                        val_malicious_idx=sampled_idx["val_malicious_idx"],
                        attack_booster=attack_booster,
                        attack_threshold=attack_threshold,
                        attack_source_mode=stability_attack_source_mode,
                        profile=profile,
                        model_hparams=hparams,
                        stage_name="stability",
                        stage_search_profile=stability_profile,
                        epsilons=epsilons,
                        gate_epsilon=gate_epsilon,
                        query_budget_benign=int(stability_profile["query_budget_benign"]),
                        query_max_steps_benign=int(stability_profile["query_max_steps_benign"]),
                        query_candidates_benign=int(stability_profile["query_candidates_benign"]),
                        query_subset_benign=int(stability_profile["query_subset_benign"]),
                        query_score_batch_benign=int(stability_profile["query_score_batch_benign"]),
                        query_fast_projection=bool(stability_profile.get("query_fast_projection", True)),
                        query_refine_topk=max(1, int(stability_profile.get("query_refine_topk", 2))),
                        query_max_active_rows_benign=stability_profile["query_max_active_rows_benign"],
                        query_stagnation_patience_benign=stability_profile["query_stagnation_patience_benign"],
                        query_stagnation_min_delta_benign=float(stability_profile["query_stagnation_min_delta_benign"]),
                        query_budget_mal=int(stability_profile["query_budget_mal"]),
                        query_max_steps_mal=int(stability_profile["query_max_steps_mal"]),
                        query_candidates_mal=int(stability_profile["query_candidates_mal"]),
                        query_subset_mal=int(stability_profile["query_subset_mal"]),
                        query_score_batch_mal=int(stability_profile["query_score_batch_mal"]),
                        query_max_active_rows_mal=stability_profile["query_max_active_rows_mal"],
                        query_stagnation_patience_mal=stability_profile["query_stagnation_patience_mal"],
                        query_stagnation_min_delta_mal=float(stability_profile["query_stagnation_min_delta_mal"]),
                        threshold_grid_size=int(args.threshold_grid_size),
                        gate_clean_fpr_max=float(args.gate_clean_fpr_max),
                        gate_attacked_benign_fpr_max=float(args.gate_attacked_benign_fpr_max),
                        gate_adv_malicious_recall_min=float(args.gate_adv_malicious_recall_min),
                        threshold_gate_attacked_benign_margin=float(args.threshold_gate_attacked_benign_margin),
                        strict_fpr_feasibility_check=bool(strict_fpr_feasibility_check),
                        min_val_benign_for_fpr_gate=int(min_val_benign_for_fpr_gate),
                        split_seed_used=int(pdata.get("split_seed_used", stability_split_seed)),
                        xgb_device=str(args.xgb_device),
                        mlp_epochs=int(args.mlp_epochs),
                        mlp_predict_batch_size=int(args.mlp_predict_batch_size),
                        bluetooth_hardneg_max_fraction=float(bluetooth_hardneg_max_fraction),
                        preview_max_rows=int(args.preview_max_rows),
                        save_models=bool(args.save_models),
                        start_ts=start_ts,
                    )
                    seed_proto_rows.append(row)
            seed_global_rows = _build_global_rows(protocol_rows=seed_proto_rows, protocols=protocols)
            for gr in seed_global_rows:
                stability_rows.append(
                    {
                        "seed": int(seed),
                        "split_seed_used": int(stability_split_seed),
                        "stability_sampling_mode": str(stability_sampling_mode),
                        "candidate_key": str(gr["candidate_key"]),
                        "candidate_group_key": str(gr.get("candidate_group_key", f"{gr['model_name']}__{gr['family_id']}")),
                        "model_name": str(gr["model_name"]),
                        "family_id": str(gr["family_id"]),
                        "global_gate_pass": bool(gr["gate_pass"]),
                        "worst_attacked_benign_fpr": float(gr["worst_attacked_benign_fpr"]),
                        "worst_clean_fpr": float(gr["worst_clean_fpr"]),
                        "worst_adv_malicious_recall": float(gr["worst_adv_malicious_recall"]),
                        "primary_fail_reason": str(gr.get("gate_failure_category", "none")),
                    }
                )

    stability_csv = out_run_dir / "stability_check_global.csv"
    pd.DataFrame(stability_rows).to_csv(stability_csv, index=False)

    coarse_group_pass: Dict[str, bool] = {}
    for row in global_rows:
        gk = str(row.get("candidate_group_key", f"{row['model_name']}__{row['family_id']}"))
        coarse_group_pass[gk] = bool(coarse_group_pass.get(gk, False) or bool(row.get("gate_pass", False)))

    stability_summary_rows: List[Dict[str, Any]] = []
    stable_accept = False
    if stability_rows:
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for row in stability_rows:
            gk = str(row.get("candidate_group_key", f"{row['model_name']}__{row['family_id']}"))
            by_group.setdefault(gk, []).append(row)
        for group_key, rows in by_group.items():
            pass_list = [bool(coarse_group_pass.get(group_key, False))] + [bool(r["global_gate_pass"]) for r in rows]
            consistent = all(pass_list)
            stable_accept = stable_accept or bool(consistent)
            stability_summary_rows.append(
                {
                    "candidate_group_key": group_key,
                    "model_name": str(rows[0].get("model_name", "")),
                    "family_id": str(rows[0].get("family_id", "")),
                    "consistent_gate_pass": bool(consistent),
                    "num_seeds_checked": int(len(pass_list)),
                    "any_seed_pass": bool(any(pass_list)),
                }
            )
    else:
                                                                                                
        stable_accept = any(bool(r.get("gate_pass", False)) for r in global_rows)

    stability_summary_csv = out_run_dir / "stability_consistency_summary.csv"
    pd.DataFrame(stability_summary_rows).to_csv(stability_summary_csv, index=False)

    protocol_hard_cap: Dict[str, bool] = {}
    for proto in protocols:
        rows = [r for r in protocol_rows if protocol_slug(str(r.get("protocol", ""))) == proto]
        protocol_hard_cap[proto] = not any(int(r.get("num_gate_pass", 0)) > 0 for r in rows)

    tuning_infeasible = any(bool(v) for v in protocol_hard_cap.values())
    stability_infeasible = bool(stability_rows) and (not stable_accept) and any(
        bool(r.get("gate_pass", False)) for r in global_rows
    )
    data_cap_infeasible = (
        any("data_cap_infeasible" in str(r.get("gate_failure_category", "")) for r in protocol_rows)
        or any("data_cap_infeasible" in str(r.get("gate_failure_category", "")) for r in global_rows)
        or any("data_cap_infeasible" in str(r.get("primary_fail_reason", "")) for r in stability_rows)
    )
    escalate = not stable_accept
    if not escalate:
        reason = "none"
        recommendation = "stable_candidate_ready"
    elif data_cap_infeasible:
        reason = "data_cap_infeasible"
        recommendation = "increase_validation_support_or_adjust_split_policy"
    elif tuning_infeasible:
        reason = "tuning_infeasible"
        recommendation = "stop_tuning_loop_move_to_representation_changes"
    elif stability_infeasible:
        reason = "stability_infeasible"
        recommendation = "stop_tuning_loop_move_to_representation_changes"
    else:
        reason = "no_stable_candidate"
        recommendation = "consider_representation_changes"

    escalation = {
        "triggered": bool(escalate),
        "reason": str(reason),
        "recommendation": str(recommendation),
        "protocol_hard_cap": protocol_hard_cap,
        "strict_fpr_feasibility_check": bool(strict_fpr_feasibility_check),
        "min_val_benign_for_fpr_gate": int(min_val_benign_for_fpr_gate),
    }
    escalation_json = out_run_dir / "escalation_recommendation.json"
    with escalation_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(escalation), f, indent=2)

    constraints_summary_path = out_run_dir / "hardening_constraints_summary.csv"
    constraints_summary_df.to_csv(constraints_summary_path, index=False)
    realism_profile_path = out_run_dir / "hardening_realism_profile.json"
    with realism_profile_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(realism_profile_json), f, indent=2)

    data_cap_summary_csv = out_run_dir / "data_cap_summary.csv"
    data_cap_df = pd.DataFrame(all_data_cap_rows)
    if not data_cap_df.empty:
        data_cap_df = data_cap_df.sort_values(
            by=["split_seed_used", "protocol", "label"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    data_cap_df.to_csv(data_cap_summary_csv, index=False)

    ext_json = out_run_dir / "external_benign_data_summary.json"
    with ext_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(external_info), f, indent=2)
    pd.DataFrame([external_info]).to_csv(out_run_dir / "external_benign_data_summary.csv", index=False)

    notice_path = out_run_dir / "AUTHORITATIVE_RESULTS_NOTICE.txt"
    notice_path.write_text(
        (
            "Authoritative: decision_table_protocol_<protocol>.csv, decision_table_global.csv,\n"
            "stability_check_global.csv, escalation_recommendation.json.\n"
            "Legacy copied metrics are non-authoritative.\n"
        ),
        encoding="utf-8",
    )

    matrix_summary = {
        "generated_at": datetime.now().isoformat(),
        "base_run_dir": str(base_run_dir),
        "output_run_dir": str(out_run_dir),
        "stage_mode": str(args.stage_mode),
        "protocols": protocols,
        "models": models,
        "families": to_jsonable(family_cfgs),
        "attack_profile": {
            "mode": "fixed_worst_case_query",
            "fixed_reference_model": "xgboost_from_base_run",
            "attack_source_mode_requested": str(requested_attack_source_mode),
            "attack_source_mode_coarse": str(
                _resolve_attack_source_mode_for_stage(requested_attack_source_mode, "coarse")
            ),
            "attack_source_mode_stability": str(
                _resolve_attack_source_mode_for_stage(requested_attack_source_mode, "stability")
            ),
            "stability_split_mode": str(stability_split_mode),
            "shared_split_seed": int(shared_split_seed),
            "stability_sampling_mode": str(stability_sampling_mode),
            "shared_sampling_seed": int(shared_sampling_seed),
            "family_pack": str(args.family_pack),
            "bluetooth_hardneg_max_fraction": float(bluetooth_hardneg_max_fraction),
            "bluetooth_benign_min_train_rows": int(bluetooth_benign_min_train_rows),
            "epsilons": [float(e) for e in epsilons],
            "gate_epsilon": float(gate_epsilon),
            "coarse_models": [str(m) for m in coarse_models],
            "stability_models_effective": sorted({str(spec.model_name) for spec in shortlist_specs}),
            "coarse_search_profile": to_jsonable(coarse_profile),
            "stability_search_profile": to_jsonable(stability_profile),
        },
        "gates": {
            "clean_fpr_max": float(args.gate_clean_fpr_max),
            "attacked_benign_fpr_max": float(args.gate_attacked_benign_fpr_max),
            "adv_malicious_recall_min": float(args.gate_adv_malicious_recall_min),
            "threshold_gate_attacked_benign_margin": float(args.threshold_gate_attacked_benign_margin),
            "strict_fpr_feasibility_check": bool(strict_fpr_feasibility_check),
            "min_val_benign_for_fpr_gate": int(min_val_benign_for_fpr_gate),
        },
        "protocol_rows": to_jsonable(per_proto_tables),
        "global_rows": to_jsonable(global_rows),
        "stability_rows": to_jsonable(stability_rows),
        "stability_consistency_rows": to_jsonable(stability_summary_rows),
        "fallback_events": to_jsonable(all_fallback_events),
        "data_cap_rows": to_jsonable(all_data_cap_rows),
        "escalation_recommendation": escalation,
        "external_benign_summary": to_jsonable(external_info),
        "files": {
            "decision_table_global_csv": str(global_csv),
            "stability_check_global_csv": str(stability_csv),
            "stability_consistency_csv": str(stability_summary_csv),
            "escalation_recommendation_json": str(escalation_json),
            "constraints_summary_csv": str(constraints_summary_path),
            "realism_profile_json": str(realism_profile_path),
            "data_cap_summary_csv": str(data_cap_summary_csv),
            "authoritative_notice": str(notice_path),
        },
    }
    matrix_summary_path = out_run_dir / "matrix_summary.json"
    with matrix_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(matrix_summary), f, indent=2)

    log_progress(f"saved protocol multimodel matrix run: {out_run_dir}", start_ts=start_ts)
    log_progress(f"saved global decision table: {global_csv}", start_ts=start_ts)
    log_progress(f"saved data cap summary: {data_cap_summary_csv}", start_ts=start_ts)
    log_progress(f"saved escalation recommendation: {escalation_json}", start_ts=start_ts)
    print(str(out_run_dir))

if __name__ == "__main__":
    main_protocol_multimodel()
