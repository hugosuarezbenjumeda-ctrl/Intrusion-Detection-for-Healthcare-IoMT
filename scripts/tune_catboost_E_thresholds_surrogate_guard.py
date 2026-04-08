                      

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluate_catboost_E_surrogate_attacks import (
    attack_malicious_subset,
    choose_default_protocol,
    load_catboost_bundle,
    parse_attack_methods,
    parse_protocols,
)
from evaluate_xgb_robustness import (
    build_constraints,
    log_progress,
    parse_epsilons,
    sample_train_rows_by_protocol,
    to_jsonable,
    train_surrogates_by_protocol,
)
from xgb_protocol_ids_utils import load_feature_columns, prepare_feature_matrix, protocol_slug


def parse_float_csv(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("Empty float list.")
    return out


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return float("nan")
    return float(num) / float(den)


def counts_from_scores(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    pred = scores >= float(threshold)
    y = y_true.astype(np.uint8, copy=False)
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & (~pred)))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred)))
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 0.0
    if not math.isnan(precision) and not math.isnan(recall) and float(precision + recall) > 0.0:
        f1 = float((2.0 * precision * recall) / (precision + recall))
    fpr = safe_div(fp, fp + tn)
    return {
        "precision": float(precision) if not math.isnan(precision) else float("nan"),
        "recall": float(recall) if not math.isnan(recall) else float("nan"),
        "f1": float(f1),
        "fpr": float(fpr) if not math.isnan(fpr) else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(tp + tn + fp + fn),
    }


def add_counts(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def find_threshold_index(values: Sequence[float], target: float) -> int:
    vals = np.asarray(values, dtype=np.float64)
    idx = int(np.argmin(np.abs(vals - float(target))))
    if float(abs(vals[idx] - float(target))) > 1e-10:
        raise RuntimeError(f"Could not find threshold index for target={target}")
    return idx


def load_test_protocol_arrays(
    *,
    test_csv: Path,
    feature_columns: Sequence[str],
    protocols: Sequence[str],
    default_protocol: str,
    chunk_size: int,
    max_test_rows: int,
    start_ts: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    by_proto_x: Dict[str, List[np.ndarray]] = {str(p): [] for p in protocols}
    by_proto_y: Dict[str, List[np.ndarray]] = {str(p): [] for p in protocols}
    rows_scanned = 0
    chunks_seen = 0

    usecols = ["protocol_hint", "label"] + list(feature_columns)
    reader = pd.read_csv(test_csv, usecols=usecols, chunksize=max(1, int(chunk_size)))
    for chunk in reader:
        if int(max_test_rows) > 0 and rows_scanned >= int(max_test_rows):
            break
        if int(max_test_rows) > 0:
            remaining = int(max_test_rows) - int(rows_scanned)
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        chunks_seen += 1
        rows_scanned += int(len(chunk))

        raw_proto = chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
        routed_proto = np.array(
            [p if p in by_proto_x else str(default_protocol) for p in raw_proto],
            dtype=object,
        )
        y = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(np.uint8).clip(0, 1).to_numpy(copy=False)
        x = prepare_feature_matrix(chunk.copy(), list(feature_columns))

        for proto in protocols:
            mask = routed_proto == str(proto)
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            by_proto_x[str(proto)].append(x[idx].astype(np.float32, copy=False))
            by_proto_y[str(proto)].append(y[idx].astype(np.uint8, copy=False))

        if chunks_seen % 2 == 0:
            log_progress(
                f"test load progress: chunks={chunks_seen}, rows={rows_scanned}",
                start_ts=start_ts,
            )

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for proto in protocols:
        p = str(proto)
        if not by_proto_x[p]:
            out[p] = {
                "x": np.empty((0, len(feature_columns)), dtype=np.float32),
                "y": np.empty((0,), dtype=np.uint8),
            }
            continue
        out[p] = {
            "x": np.concatenate(by_proto_x[p], axis=0),
            "y": np.concatenate(by_proto_y[p], axis=0),
        }
    log_progress(f"test load complete: rows={rows_scanned}, chunks={chunks_seen}", start_ts=start_ts)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851"),
    )
    ap.add_argument(
        "--matrix-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105"),
    )
    ap.add_argument("--train-csv", type=Path, default=Path("data/merged/metadata_train.csv"))
    ap.add_argument("--test-csv", type=Path, default=Path("data/merged/metadata_test.csv"))
    ap.add_argument("--protocols", type=str, default="wifi,mqtt,bluetooth")
    ap.add_argument("--model-name", type=str, default="catboost")
    ap.add_argument("--family-id", type=str, default="E")
    ap.add_argument("--stage-name", type=str, choices=("coarse", "stability"), default="coarse")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--attack-methods", type=str, default="surrogate_fgsm,surrogate_pgd")
    ap.add_argument("--epsilons", type=str, default="0,0.01,0.02,0.05,0.10")
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--pgd-alpha-ratio", type=float, default=0.25)

    ap.add_argument("--surrogate-train-per-protocol", type=int, default=200000)
    ap.add_argument("--surrogate-epochs", type=int, default=12)
    ap.add_argument("--surrogate-lr", type=float, default=0.08)
    ap.add_argument("--surrogate-batch-size", type=int, default=4096)
    ap.add_argument("--percentile-lower", type=float, default=1.0)
    ap.add_argument("--percentile-upper", type=float, default=99.0)

    ap.add_argument(
        "--threshold-deltas",
        type=str,
        default="-0.02,-0.01,-0.005,0,0.005,0.01,0.02",
        help="Small per-protocol deltas around each current threshold.",
    )

    ap.add_argument(
        "--max-clean-fpr-increase",
        type=float,
        default=0.0,
        help="Absolute allowed increase vs baseline clean global FPR.",
    )
    ap.add_argument(
        "--max-attacked-benign-fpr-increase",
        type=float,
        default=0.0,
        help="Absolute allowed increase vs baseline worst attacked-benign FPR.",
    )
    ap.add_argument(
        "--max-adv-recall-drop",
        type=float,
        default=0.002,
        help="Allowed worst-case adversarial recall drop vs baseline.",
    )
    ap.add_argument(
        "--max-robust-f1-drop",
        type=float,
        default=0.002,
        help="Allowed worst-case robust F1 drop vs baseline.",
    )

    ap.add_argument("--chunk-size", type=int, default=200000)
    ap.add_argument("--max-test-rows", type=int, default=0)

    ap.add_argument(
        "--output-prefix",
        type=str,
        default="catboost_E_threshold_tiny_tune_surrogate_guard",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    start_ts = time.time()

    base_run_dir = args.base_run_dir.resolve()
    matrix_run_dir = args.matrix_run_dir.resolve()
    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()

    protocols = parse_protocols(args.protocols)
    attack_methods = parse_attack_methods(args.attack_methods)
    epsilons = parse_epsilons(args.epsilons)
    threshold_deltas = sorted(set(parse_float_csv(args.threshold_deltas)))

    if 0.0 not in threshold_deltas:
        threshold_deltas.append(0.0)
        threshold_deltas = sorted(set(threshold_deltas))

    feature_columns = load_feature_columns(base_run_dir)
    default_protocol = choose_default_protocol(protocols)

    log_progress("loading finalized CatBoost E artifacts", start_ts=start_ts)
    bundles: Dict[str, Dict[str, Any]] = {}
    base_thresholds: Dict[str, float] = {}
    for proto in protocols:
        bundle = load_catboost_bundle(
            matrix_run_dir=matrix_run_dir,
            protocol=str(proto),
            model_name=str(args.model_name),
            family_id=str(args.family_id),
            seed=int(args.seed),
            stage_name=str(args.stage_name),
        )
        bundles[str(proto)] = bundle
        base_thresholds[str(proto)] = float(bundle["threshold"])

    threshold_grid: Dict[str, List[float]] = {}
    for proto in protocols:
        base_thr = float(base_thresholds[str(proto)])
        vals: List[float] = []
        for d in threshold_deltas:
            t = min(1.0, max(0.0, base_thr + float(d)))
            vals.append(float(t))
        vals = sorted(set(vals + [base_thr]))
        threshold_grid[str(proto)] = vals

    log_progress("sampling train data for surrogate fitting", start_ts=start_ts)
    train_sample_df, train_sample_meta = sample_train_rows_by_protocol(
        train_csv=train_csv,
        feature_columns=feature_columns,
        target_per_protocol=int(args.surrogate_train_per_protocol),
        protocols=protocols,
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )

    log_progress("building constraints and surrogates", start_ts=start_ts)
    constraints, _ = build_constraints(
        train_sample_df=train_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        percentile_lower=float(args.percentile_lower),
        percentile_upper=float(args.percentile_upper),
    )
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

    log_progress("loading full test rows by protocol", start_ts=start_ts)
    test_by_proto = load_test_protocol_arrays(
        test_csv=test_csv,
        feature_columns=feature_columns,
        protocols=protocols,
        default_protocol=default_protocol,
        chunk_size=int(args.chunk_size),
        max_test_rows=int(args.max_test_rows),
        start_ts=start_ts,
    )

    scenario_keys: List[Tuple[str, float]] = [("baseline", 0.0)]
    for method in attack_methods:
        for eps in epsilons:
            scenario_keys.append((str(method), float(eps)))

    attack_eval_keys: List[Tuple[str, float]] = [
        (str(m), float(e))
        for m in attack_methods
        for e in epsilons
        if float(e) > 0.0
    ]
    if not attack_eval_keys:
        raise RuntimeError("Need at least one epsilon > 0 for robustness guard evaluation.")

    log_progress("precomputing clean + attacked scores", start_ts=start_ts)
    score_cache: Dict[str, Dict[Tuple[str, float], np.ndarray]] = {}
    for proto in protocols:
        p = str(proto)
        pdata = test_by_proto[p]
        x = pdata["x"]
        y = pdata["y"]
        bundle = bundles[p]
        model = bundle["model"]

        if x.shape[0] == 0:
            raise RuntimeError(f"No routed test rows for protocol '{p}'.")

        clean_scores = model.predict_proba(x)[:, 1].astype(np.float32, copy=False)
        score_map: Dict[Tuple[str, float], np.ndarray] = {
            ("baseline", 0.0): clean_scores,
        }

        mal_mask = y == 1
        x_mal = x[mal_mask] if np.any(mal_mask) else np.empty((0, x.shape[1]), dtype=np.float32)
        lower = constraints[p]["lower"].astype(np.float64)
        upper = constraints[p]["upper"].astype(np.float64)
        locked_idx = constraints[p]["locked_idx"]

        for method in attack_methods:
            for eps in epsilons:
                key = (str(method), float(eps))
                if float(eps) <= 0.0 or x_mal.size == 0:
                    score_map[key] = clean_scores
                    continue

                x_adv_mal = attack_malicious_subset(
                    method=str(method),
                    epsilon=float(eps),
                    x_malicious=x_mal,
                    surrogate=surrogates[p],
                    lower=lower,
                    upper=upper,
                    locked_idx=locked_idx,
                    pgd_steps=int(args.pgd_steps),
                    pgd_alpha_ratio=float(args.pgd_alpha_ratio),
                )
                adv_scores_mal = model.predict_proba(x_adv_mal)[:, 1].astype(np.float32, copy=False)
                adv_scores = clean_scores.copy()
                adv_scores[mal_mask] = adv_scores_mal
                score_map[key] = adv_scores

        score_cache[p] = score_map
        log_progress(f"scores ready for protocol={p}, rows={x.shape[0]}", start_ts=start_ts)

    log_progress("precomputing confusion counts per threshold", start_ts=start_ts)
    precomputed: Dict[str, Dict[Tuple[str, float], List[Tuple[int, int, int, int]]]] = {}
    for proto in protocols:
        p = str(proto)
        y = test_by_proto[p]["y"]
        precomputed[p] = {}
        for scenario in scenario_keys:
            scores = score_cache[p][scenario]
            counts_list: List[Tuple[int, int, int, int]] = []
            for thr in threshold_grid[p]:
                counts_list.append(counts_from_scores(scores=scores, y_true=y, threshold=float(thr)))
            precomputed[p][scenario] = counts_list

    base_idx: Dict[str, int] = {
        str(proto): find_threshold_index(threshold_grid[str(proto)], float(base_thresholds[str(proto)]))
        for proto in protocols
    }

    def evaluate_combo(idx_map: Dict[str, int]) -> Dict[str, Any]:
        clean_counts = (0, 0, 0, 0)
        for proto in protocols:
            p = str(proto)
            clean_counts = add_counts(clean_counts, precomputed[p][("baseline", 0.0)][idx_map[p]])
        clean_m = metrics_from_counts(*clean_counts)

        worst_attacked_benign_fpr = float("-inf")
        worst_adv_malicious_recall = float("inf")
        worst_robust_f1 = float("inf")
        worst_attacked_key = ""
        worst_recall_key = ""
        worst_f1_key = ""

        attack_rows: List[Dict[str, Any]] = []
        for scenario in attack_eval_keys:
            att_counts = (0, 0, 0, 0)
            for proto in protocols:
                p = str(proto)
                att_counts = add_counts(att_counts, precomputed[p][scenario][idx_map[p]])
            att_m = metrics_from_counts(*att_counts)

            attacked_benign_fpr = float(att_m["fpr"])
            adv_recall = float(att_m["recall"])
            robust_f1 = float(att_m["f1"])

            if (not math.isnan(attacked_benign_fpr)) and attacked_benign_fpr > worst_attacked_benign_fpr:
                worst_attacked_benign_fpr = attacked_benign_fpr
                worst_attacked_key = f"{scenario[0]}@eps={scenario[1]}"
            if (not math.isnan(adv_recall)) and adv_recall < worst_adv_malicious_recall:
                worst_adv_malicious_recall = adv_recall
                worst_recall_key = f"{scenario[0]}@eps={scenario[1]}"
            if (not math.isnan(robust_f1)) and robust_f1 < worst_robust_f1:
                worst_robust_f1 = robust_f1
                worst_f1_key = f"{scenario[0]}@eps={scenario[1]}"

            attack_rows.append(
                {
                    "attack_method": str(scenario[0]),
                    "epsilon": float(scenario[1]),
                    "precision": float(att_m["precision"]),
                    "adv_malicious_recall": float(att_m["recall"]),
                    "robust_f1": float(att_m["f1"]),
                    "attacked_benign_fpr": float(att_m["fpr"]),
                    "tp": int(att_m["tp"]),
                    "tn": int(att_m["tn"]),
                    "fp": int(att_m["fp"]),
                    "fn": int(att_m["fn"]),
                }
            )

        if worst_attacked_benign_fpr == float("-inf"):
            worst_attacked_benign_fpr = float("nan")
        if worst_adv_malicious_recall == float("inf"):
            worst_adv_malicious_recall = float("nan")
        if worst_robust_f1 == float("inf"):
            worst_robust_f1 = float("nan")

        thr_map = {str(p): float(threshold_grid[str(p)][idx_map[str(p)]]) for p in protocols}
        thr_l1_shift = float(sum(abs(thr_map[str(p)] - float(base_thresholds[str(p)])) for p in protocols))

        return {
            "thresholds": thr_map,
            "threshold_l1_shift": thr_l1_shift,
            "clean_precision": float(clean_m["precision"]),
            "clean_recall": float(clean_m["recall"]),
            "clean_f1": float(clean_m["f1"]),
            "clean_fpr": float(clean_m["fpr"]),
            "clean_tp": int(clean_m["tp"]),
            "clean_tn": int(clean_m["tn"]),
            "clean_fp": int(clean_m["fp"]),
            "clean_fn": int(clean_m["fn"]),
            "worst_attacked_benign_fpr": float(worst_attacked_benign_fpr),
            "worst_adv_malicious_recall": float(worst_adv_malicious_recall),
            "worst_robust_f1": float(worst_robust_f1),
            "worst_attacked_benign_fpr_key": str(worst_attacked_key),
            "worst_adv_recall_key": str(worst_recall_key),
            "worst_robust_f1_key": str(worst_f1_key),
            "attack_rows": attack_rows,
        }

    baseline_eval = evaluate_combo(base_idx)

    log_progress("evaluating threshold combinations", start_ts=start_ts)
    proto_lists = [str(p) for p in protocols]
    idx_ranges = [range(len(threshold_grid[p])) for p in proto_lists]

    candidate_rows: List[Dict[str, Any]] = []
    attack_compare_rows: List[Dict[str, Any]] = []

    total_combos = int(np.prod([len(list(r)) for r in idx_ranges]))
    seen = 0

    for combo in itertools.product(*idx_ranges):
        idx_map = {proto_lists[i]: int(combo[i]) for i in range(len(proto_lists))}
        ev = evaluate_combo(idx_map)
        seen += 1

        pass_clean_fpr = bool(
            ev["clean_fpr"] <= baseline_eval["clean_fpr"] + float(args.max_clean_fpr_increase)
        )
        pass_attacked_benign_fpr = bool(
            ev["worst_attacked_benign_fpr"]
            <= baseline_eval["worst_attacked_benign_fpr"] + float(args.max_attacked_benign_fpr_increase)
        )
        pass_adv_recall = bool(
            ev["worst_adv_malicious_recall"]
            >= baseline_eval["worst_adv_malicious_recall"] - float(args.max_adv_recall_drop)
        )
        pass_robust_f1 = bool(
            ev["worst_robust_f1"] >= baseline_eval["worst_robust_f1"] - float(args.max_robust_f1_drop)
        )
        guard_pass = bool(pass_clean_fpr and pass_attacked_benign_fpr and pass_adv_recall and pass_robust_f1)

        row = {
            "wifi_threshold": float(ev["thresholds"].get("wifi", float("nan"))),
            "mqtt_threshold": float(ev["thresholds"].get("mqtt", float("nan"))),
            "bluetooth_threshold": float(ev["thresholds"].get("bluetooth", float("nan"))),
            "threshold_l1_shift": float(ev["threshold_l1_shift"]),
            "clean_precision": float(ev["clean_precision"]),
            "clean_recall": float(ev["clean_recall"]),
            "clean_f1": float(ev["clean_f1"]),
            "clean_fpr": float(ev["clean_fpr"]),
            "worst_attacked_benign_fpr": float(ev["worst_attacked_benign_fpr"]),
            "worst_adv_malicious_recall": float(ev["worst_adv_malicious_recall"]),
            "worst_robust_f1": float(ev["worst_robust_f1"]),
            "worst_attacked_benign_fpr_key": str(ev["worst_attacked_benign_fpr_key"]),
            "worst_adv_recall_key": str(ev["worst_adv_recall_key"]),
            "worst_robust_f1_key": str(ev["worst_robust_f1_key"]),
            "pass_clean_fpr_guard": bool(pass_clean_fpr),
            "pass_attacked_benign_fpr_guard": bool(pass_attacked_benign_fpr),
            "pass_adv_recall_guard": bool(pass_adv_recall),
            "pass_robust_f1_guard": bool(pass_robust_f1),
            "guard_pass": bool(guard_pass),
            "is_baseline_thresholds": bool(all(idx_map[p] == base_idx[p] for p in proto_lists)),
        }
        candidate_rows.append(row)

        if row["is_baseline_thresholds"]:
            for ar in ev["attack_rows"]:
                attack_compare_rows.append(
                    {
                        "policy": "baseline_thresholds",
                        **ar,
                    }
                )

        if seen % 25 == 0 or seen == total_combos:
            log_progress(f"threshold combos evaluated: {seen}/{total_combos}", start_ts=start_ts)

    eligible = [r for r in candidate_rows if bool(r.get("guard_pass", False))]
    selection_mode = "guarded"
    if not eligible:
        eligible = list(candidate_rows)
        selection_mode = "fallback_no_guard_pass"

    eligible_sorted = sorted(
        eligible,
        key=lambda r: (
            -float(r["clean_f1"]),
            -float(r["worst_robust_f1"]),
            -float(r["worst_adv_malicious_recall"]),
            float(r["clean_fpr"]),
            float(r["threshold_l1_shift"]),
        ),
    )
    best_row = dict(eligible_sorted[0])

    best_idx_map: Dict[str, int] = {}
    for proto in proto_lists:
        best_thr = float(best_row[f"{proto}_threshold"]) if f"{proto}_threshold" in best_row else float("nan")
        if math.isnan(best_thr):
            if proto == "wifi":
                best_thr = float(best_row["wifi_threshold"])
            elif proto == "mqtt":
                best_thr = float(best_row["mqtt_threshold"])
            elif proto == "bluetooth":
                best_thr = float(best_row["bluetooth_threshold"])
        best_idx_map[proto] = find_threshold_index(threshold_grid[proto], best_thr)

    tuned_eval = evaluate_combo(best_idx_map)
    for ar in tuned_eval["attack_rows"]:
        attack_compare_rows.append(
            {
                "policy": "tuned_thresholds",
                **ar,
            }
        )

    decision_ref = None
    decision_path = matrix_run_dir / "decision_table_global.csv"
    if decision_path.exists():
        try:
            dfg = pd.read_csv(decision_path)
            mask = (
                dfg["model_name"].astype(str).str.lower() == str(args.model_name).lower()
            ) & (
                dfg["family_id"].astype(str).str.upper() == str(args.family_id).upper()
            ) & (
                pd.to_numeric(dfg.get("seed", pd.Series([np.nan] * len(dfg))), errors="coerce") == int(args.seed)
            )
            pick = dfg.loc[mask].copy()
            if not pick.empty:
                decision_ref = pick.sort_values(by=["global_rank"], ascending=True).iloc[0].to_dict()
        except Exception:
            decision_ref = None

    summary_rows: List[Dict[str, Any]] = [
        {
            "policy": "baseline_thresholds",
            "wifi_threshold": float(base_thresholds.get("wifi", float("nan"))),
            "mqtt_threshold": float(base_thresholds.get("mqtt", float("nan"))),
            "bluetooth_threshold": float(base_thresholds.get("bluetooth", float("nan"))),
            "clean_precision": float(baseline_eval["clean_precision"]),
            "clean_recall": float(baseline_eval["clean_recall"]),
            "clean_f1": float(baseline_eval["clean_f1"]),
            "clean_fpr": float(baseline_eval["clean_fpr"]),
            "attacked_benign_fpr": float(baseline_eval["worst_attacked_benign_fpr"]),
            "adv_malicious_recall": float(baseline_eval["worst_adv_malicious_recall"]),
            "robust_f1": float(baseline_eval["worst_robust_f1"]),
            "worst_attacked_benign_fpr_key": str(baseline_eval["worst_attacked_benign_fpr_key"]),
            "worst_adv_recall_key": str(baseline_eval["worst_adv_recall_key"]),
            "worst_robust_f1_key": str(baseline_eval["worst_robust_f1_key"]),
            "guard_pass": True,
            "selection_mode": "reference",
        },
        {
            "policy": "tuned_thresholds",
            "wifi_threshold": float(tuned_eval["thresholds"].get("wifi", float("nan"))),
            "mqtt_threshold": float(tuned_eval["thresholds"].get("mqtt", float("nan"))),
            "bluetooth_threshold": float(tuned_eval["thresholds"].get("bluetooth", float("nan"))),
            "clean_precision": float(tuned_eval["clean_precision"]),
            "clean_recall": float(tuned_eval["clean_recall"]),
            "clean_f1": float(tuned_eval["clean_f1"]),
            "clean_fpr": float(tuned_eval["clean_fpr"]),
            "attacked_benign_fpr": float(tuned_eval["worst_attacked_benign_fpr"]),
            "adv_malicious_recall": float(tuned_eval["worst_adv_malicious_recall"]),
            "robust_f1": float(tuned_eval["worst_robust_f1"]),
            "worst_attacked_benign_fpr_key": str(tuned_eval["worst_attacked_benign_fpr_key"]),
            "worst_adv_recall_key": str(tuned_eval["worst_adv_recall_key"]),
            "worst_robust_f1_key": str(tuned_eval["worst_robust_f1_key"]),
            "guard_pass": bool(best_row.get("guard_pass", False)),
            "selection_mode": str(selection_mode),
        },
    ]

    if decision_ref is not None:
        summary_rows.append(
            {
                "policy": "pipeline_reference_decision_table_global",
                "wifi_threshold": float("nan"),
                "mqtt_threshold": float("nan"),
                "bluetooth_threshold": float("nan"),
                "clean_precision": float("nan"),
                "clean_recall": float("nan"),
                "clean_f1": float(decision_ref.get("clean_f1", float("nan"))),
                "clean_fpr": float(decision_ref.get("clean_fpr", float("nan"))),
                "attacked_benign_fpr": float(decision_ref.get("worst_attacked_benign_fpr", float("nan"))),
                "adv_malicious_recall": float(decision_ref.get("worst_adv_malicious_recall", float("nan"))),
                "robust_f1": float(decision_ref.get("robust_f1", float("nan"))),
                "worst_attacked_benign_fpr_key": "from_decision_table",
                "worst_adv_recall_key": "from_decision_table",
                "worst_robust_f1_key": "from_decision_table",
                "guard_pass": True,
                "selection_mode": "reference",
            }
        )

    out_dir = matrix_run_dir
    prefix = str(args.output_prefix).strip() or "catboost_E_threshold_tiny_tune_surrogate_guard"
    candidates_csv = out_dir / f"{prefix}_candidates.csv"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    attack_compare_csv = out_dir / f"{prefix}_attack_compare.csv"
    manifest_json = out_dir / f"{prefix}_manifest.json"

    pd.DataFrame(candidate_rows).sort_values(
        by=["guard_pass", "clean_f1", "worst_robust_f1", "worst_adv_malicious_recall", "clean_fpr"],
        ascending=[False, False, False, False, True],
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(attack_compare_rows).to_csv(attack_compare_csv, index=False)

    manifest = {
        "generated_at_epoch_sec": time.time(),
        "elapsed_sec": time.time() - start_ts,
        "selection_mode": str(selection_mode),
        "base_run_dir": str(base_run_dir),
        "matrix_run_dir": str(matrix_run_dir),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "model_name": str(args.model_name),
        "family_id": str(args.family_id),
        "stage_name": str(args.stage_name),
        "seed": int(args.seed),
        "protocols": [str(p) for p in protocols],
        "base_thresholds": base_thresholds,
        "threshold_deltas": [float(d) for d in threshold_deltas],
        "threshold_grid": threshold_grid,
        "attack_methods": [str(m) for m in attack_methods],
        "epsilons": [float(e) for e in epsilons],
        "attack_eval_keys": [f"{m}@{e}" for m, e in attack_eval_keys],
        "guards": {
            "max_clean_fpr_increase": float(args.max_clean_fpr_increase),
            "max_attacked_benign_fpr_increase": float(args.max_attacked_benign_fpr_increase),
            "max_adv_recall_drop": float(args.max_adv_recall_drop),
            "max_robust_f1_drop": float(args.max_robust_f1_drop),
        },
        "baseline": to_jsonable(summary_rows[0]),
        "tuned": to_jsonable(summary_rows[1]),
        "train_sample_meta": to_jsonable(train_sample_meta),
        "surrogate_meta": to_jsonable(surrogate_meta),
        "files": {
            "candidates_csv": str(candidates_csv),
            "summary_csv": str(summary_csv),
            "attack_compare_csv": str(attack_compare_csv),
            "manifest_json": str(manifest_json),
        },
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    log_progress(f"saved candidates: {candidates_csv}", start_ts=start_ts)
    log_progress(f"saved summary: {summary_csv}", start_ts=start_ts)
    log_progress(f"saved attack compare: {attack_compare_csv}", start_ts=start_ts)
    log_progress(f"saved manifest: {manifest_json}", start_ts=start_ts)


if __name__ == "__main__":
    main()
