                      

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from evaluate_catboost_E_surrogate_attacks import (
    attack_malicious_subset,
    parse_attack_methods,
    parse_protocols,
)
from evaluate_catboost_protocol_test_metrics import (
    average_precision_manual,
    normalize_label_series,
    pr_auc_trapezoid_manual,
    roc_auc_manual,
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


DEFAULT_PROTOCOLS = ("wifi", "mqtt", "bluetooth")


@dataclass
class CountAccumulator:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y = np.asarray(y_true, dtype=np.uint8)
        p = np.asarray(y_pred, dtype=np.uint8)
        self.tp += int(np.sum((y == 1) & (p == 1)))
        self.tn += int(np.sum((y == 0) & (p == 0)))
        self.fp += int(np.sum((y == 0) & (p == 1)))
        self.fn += int(np.sum((y == 1) & (p == 0)))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return float("nan")
    return float(num) / float(den)


def counts_to_metrics(c: CountAccumulator) -> Dict[str, float]:
    tp, tn, fp, fn = int(c.tp), int(c.tn), int(c.fp), int(c.fn)
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_default_protocol(protocols: Sequence[str]) -> str:
    for p in protocols:
        if str(p) == "wifi":
            return "wifi"
    return str(protocols[0])


def load_catboost_bundle(
    matrix_run_dir: Path,
    protocol: str,
    model_name: str,
    family_id: str,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    stage = str(stage_name).strip().lower()
    if stage == "coarse":
        candidate_dir = (
            matrix_run_dir
            / "candidates"
            / f"coarse_{protocol}_{model_name}_{family_id}_seed_{int(seed)}"
        )
    elif stage == "stability":
        candidate_dir = matrix_run_dir / "stability" / f"seed_{int(seed)}_{protocol}_{model_name}_{family_id}"
    else:
        raise ValueError(f"Unsupported stage-name '{stage_name}'.")

    summary_path = candidate_dir / "candidate_summary.json"
    model_path = candidate_dir / "models" / f"{protocol}__{model_name}__{family_id}.cbm"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing candidate summary: {summary_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    return {
        "protocol": str(protocol),
        "threshold": float(summary["selected_threshold"]),
        "summary": summary,
        "model": model,
        "summary_path": summary_path,
        "model_path": model_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--protocols", type=str, default=",".join(DEFAULT_PROTOCOLS))
    ap.add_argument("--model-name", type=str, default="catboost")
    ap.add_argument("--family-id", type=str, default="E")
    ap.add_argument("--stage-name", type=str, choices=("coarse", "stability"), default="coarse")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--tuned-summary-csv",
        type=Path,
        default=Path(
            "reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/"
            "catboost_E_threshold_tiny_tune_surrogate_guard_summary.csv"
        ),
    )
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

    ap.add_argument("--chunk-size", type=int, default=200000)
    ap.add_argument("--max-test-rows", type=int, default=0)

    ap.add_argument(
        "--output-prefix",
        type=str,
        default="catboost_E_tuned_tables",
    )
    return ap


def dataframe_to_markdown_plain(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines: List[str] = []
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            v = row[col]
            if pd.isna(v):
                vals.append("n/a")
            elif isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        body_lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body_lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    start_ts = time.time()

    base_run_dir = args.base_run_dir.resolve()
    matrix_run_dir = args.matrix_run_dir.resolve()
    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()
    tuned_summary_csv = args.tuned_summary_csv.resolve()

    protocols = parse_protocols(args.protocols)
    attack_methods = parse_attack_methods(args.attack_methods)
    epsilons = parse_epsilons(args.epsilons)

    feature_columns = load_feature_columns(base_run_dir)

    log_progress("loading CatBoost E model artifacts", start_ts=start_ts)
    bundles: Dict[str, Dict[str, Any]] = {}
    for proto in protocols:
        bundles[str(proto)] = load_catboost_bundle(
            matrix_run_dir=matrix_run_dir,
            protocol=str(proto),
            model_name=str(args.model_name),
            family_id=str(args.family_id),
            seed=int(args.seed),
            stage_name=str(args.stage_name),
        )

                                                                               
    thresholds = {str(p): float(bundles[str(p)]["threshold"]) for p in protocols}
    if tuned_summary_csv.exists():
        tdf = pd.read_csv(tuned_summary_csv)
        row = tdf.loc[tdf["policy"].astype(str) == "tuned_thresholds"]
        if not row.empty:
            r = row.iloc[0]
            if "wifi_threshold" in r.index:
                thresholds["wifi"] = float(r["wifi_threshold"])
            if "mqtt_threshold" in r.index:
                thresholds["mqtt"] = float(r["mqtt_threshold"])
            if "bluetooth_threshold" in r.index:
                thresholds["bluetooth"] = float(r["bluetooth_threshold"])

    log_progress("sampling train rows for surrogate fitting", start_ts=start_ts)
    train_sample_df, train_sample_meta = sample_train_rows_by_protocol(
        train_csv=train_csv,
        feature_columns=feature_columns,
        target_per_protocol=int(args.surrogate_train_per_protocol),
        protocols=protocols,
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        start_ts=start_ts,
    )

    log_progress("building constraints + surrogates", start_ts=start_ts)
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

    clean_scores_by_proto: Dict[str, List[np.ndarray]] = {str(p): [] for p in protocols}
    clean_y_by_proto: Dict[str, List[np.ndarray]] = {str(p): [] for p in protocols}

                             
    baseline_counts_proto: Dict[str, CountAccumulator] = {str(p): CountAccumulator() for p in protocols}
    baseline_counts_global = CountAccumulator()

    scenario_counts_proto: Dict[Tuple[str, float, str], CountAccumulator] = {}
    scenario_counts_global: Dict[Tuple[str, float], CountAccumulator] = {}
    for m in attack_methods:
        for e in epsilons:
            if float(e) <= 0.0:
                continue
            key_g = (str(m), float(e))
            scenario_counts_global[key_g] = CountAccumulator()
            for p in protocols:
                scenario_counts_proto[(str(m), float(e), str(p))] = CountAccumulator()

    default_protocol = choose_default_protocol(protocols)
    usecols = ["protocol_hint", "label"] + feature_columns

    log_progress("running clean + attacked evaluation for tables", start_ts=start_ts)
    rows_scanned = 0
    chunks_seen = 0
    reader = pd.read_csv(test_csv, usecols=usecols, chunksize=max(1, int(args.chunk_size)))
    for chunk in reader:
        if int(args.max_test_rows) > 0 and rows_scanned >= int(args.max_test_rows):
            break
        if int(args.max_test_rows) > 0:
            remain = int(args.max_test_rows) - int(rows_scanned)
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()

        if chunk.empty:
            continue

        chunks_seen += 1
        rows_scanned += int(len(chunk))

        raw_proto = chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
        routed = np.array([p if p in bundles else default_protocol for p in raw_proto], dtype=object)
        y_all = normalize_label_series(chunk["label"])
        x_all = prepare_feature_matrix(chunk.copy(), feature_columns)

        for proto in protocols:
            p = str(proto)
            mask = routed == p
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            x = x_all[idx]
            y = y_all[idx].astype(np.uint8, copy=False)
            thr = float(thresholds[p])
            model = bundles[p]["model"]

            clean_scores = model.predict_proba(x)[:, 1].astype(np.float32, copy=False)
            clean_pred = (clean_scores >= thr).astype(np.uint8, copy=False)

            clean_scores_by_proto[p].append(clean_scores)
            clean_y_by_proto[p].append(y)

            baseline_counts_proto[p].update(y, clean_pred)
            baseline_counts_global.update(y, clean_pred)

            mal_mask = y == 1
            x_mal = x[mal_mask] if np.any(mal_mask) else np.empty((0, x.shape[1]), dtype=np.float32)
            lower = constraints[p]["lower"].astype(np.float64)
            upper = constraints[p]["upper"].astype(np.float64)
            locked_idx = constraints[p]["locked_idx"]

            for m in attack_methods:
                for e in epsilons:
                    eps = float(e)
                    if eps <= 0.0:
                        continue

                    if x_mal.size == 0:
                        adv_pred = clean_pred
                    else:
                        x_adv_mal = attack_malicious_subset(
                            method=str(m),
                            epsilon=eps,
                            x_malicious=x_mal,
                            surrogate=surrogates[p],
                            lower=lower,
                            upper=upper,
                            locked_idx=locked_idx,
                            pgd_steps=int(args.pgd_steps),
                            pgd_alpha_ratio=float(args.pgd_alpha_ratio),
                        )
                        adv_scores_mal = model.predict_proba(x_adv_mal)[:, 1].astype(np.float32, copy=False)
                        adv_pred = clean_pred.copy()
                        adv_pred[mal_mask] = (adv_scores_mal >= thr).astype(np.uint8, copy=False)

                    scenario_counts_proto[(str(m), eps, p)].update(y, adv_pred)
                    scenario_counts_global[(str(m), eps)].update(y, adv_pred)

        if chunks_seen % 2 == 0:
            log_progress(f"processed chunks={chunks_seen}, rows={rows_scanned}", start_ts=start_ts)

                                              
    table1_rows: List[Dict[str, Any]] = []
    global_y_list: List[np.ndarray] = []
    global_score_list: List[np.ndarray] = []
    global_pred_list: List[np.ndarray] = []

    for proto in protocols:
        p = str(proto)
        y = np.concatenate(clean_y_by_proto[p], axis=0) if clean_y_by_proto[p] else np.empty(0, dtype=np.uint8)
        s = np.concatenate(clean_scores_by_proto[p], axis=0) if clean_scores_by_proto[p] else np.empty(0, dtype=np.float32)
        thr = float(thresholds[p])
        pred = (s >= thr).astype(np.uint8, copy=False)

        m = counts_to_metrics(baseline_counts_proto[p])
        roc_auc = roc_auc_manual(y, s)
        pr_auc = average_precision_manual(y, s)

        table1_rows.append(
            {
                "protocol_model": f"{p}_catboost_E",
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "F1": m["f1"],
                "FPR": m["fpr"],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "tp": int(m["tp"]),
                "tn": int(m["tn"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
            }
        )

        global_y_list.append(y)
        global_score_list.append(s)
        global_pred_list.append(pred)

    gy = np.concatenate(global_y_list, axis=0)
    gs = np.concatenate(global_score_list, axis=0)
    gp = np.concatenate(global_pred_list, axis=0)
    g_counts = counts_to_metrics(baseline_counts_global)

    table1_rows.append(
        {
            "protocol_model": "Global",
            "threshold": "Model Specific",
            "precision": g_counts["precision"],
            "recall": g_counts["recall"],
            "F1": g_counts["f1"],
            "FPR": g_counts["fpr"],
            "roc_auc": roc_auc_manual(gy, gs),
            "pr_auc": average_precision_manual(gy, gs),
            "tp": int(g_counts["tp"]),
            "tn": int(g_counts["tn"]),
            "fp": int(g_counts["fp"]),
            "fn": int(g_counts["fn"]),
        }
    )

                                                
    table2_rows: List[Dict[str, Any]] = []

    def worst_case_for_proto(proto: str) -> Dict[str, float]:
        worst_recall = float("inf")
        worst_f1 = float("inf")
        for m in attack_methods:
            for e in epsilons:
                eps = float(e)
                if eps <= 0.0:
                    continue
                cm = counts_to_metrics(scenario_counts_proto[(str(m), eps, str(proto))])
                if not math.isnan(float(cm["recall"])) and float(cm["recall"]) < worst_recall:
                    worst_recall = float(cm["recall"])
                if not math.isnan(float(cm["f1"])) and float(cm["f1"]) < worst_f1:
                    worst_f1 = float(cm["f1"])
        if worst_recall == float("inf"):
            worst_recall = float("nan")
        if worst_f1 == float("inf"):
            worst_f1 = float("nan")
        return {
            "attacked_benign_fpr": float("nan"),
            "adv_malicious_recall": worst_recall,
            "robust_f1": worst_f1,
        }

    for proto in protocols:
        p = str(proto)
        clean_m = counts_to_metrics(baseline_counts_proto[p])
        worst = worst_case_for_proto(p)
        table2_rows.append(
            {
                "Protocol": p.capitalize() if p != "wifi" else "WiFi",
                "clean_f1": clean_m["f1"],
                "clean_fpr": clean_m["fpr"],
                "attacked_benign_fpr": worst["attacked_benign_fpr"],
                "adv_malicious_recall": worst["adv_malicious_recall"],
                "robust_f1": worst["robust_f1"],
                "selected_threshold": float(thresholds[p]),
            }
        )

    global_worst_recall = float("inf")
    global_worst_f1 = float("inf")
    for m in attack_methods:
        for e in epsilons:
            eps = float(e)
            if eps <= 0.0:
                continue
            cm = counts_to_metrics(scenario_counts_global[(str(m), eps)])
            if not math.isnan(float(cm["recall"])) and float(cm["recall"]) < global_worst_recall:
                global_worst_recall = float(cm["recall"])
            if not math.isnan(float(cm["f1"])) and float(cm["f1"]) < global_worst_f1:
                global_worst_f1 = float(cm["f1"])

    if global_worst_recall == float("inf"):
        global_worst_recall = float("nan")
    if global_worst_f1 == float("inf"):
        global_worst_f1 = float("nan")

    g_clean = counts_to_metrics(baseline_counts_global)
    table2_rows.append(
        {
            "Protocol": "Global",
            "clean_f1": g_clean["f1"],
            "clean_fpr": g_clean["fpr"],
            "attacked_benign_fpr": float("nan"),
            "adv_malicious_recall": global_worst_recall,
            "robust_f1": global_worst_f1,
            "selected_threshold": "Model Specific",
        }
    )

    out_dir = matrix_run_dir
    prefix = str(args.output_prefix).strip() or "catboost_E_tuned_tables"

    table1_csv = out_dir / f"{prefix}_table_5_1_protocol_metrics.csv"
    table2_csv = out_dir / f"{prefix}_table_5_2_robustness_metrics.csv"
    table_md = out_dir / f"{prefix}_thesis_tables.md"
    manifest_json = out_dir / f"{prefix}_manifest.json"

    t1_df = pd.DataFrame(table1_rows)
    t2_df = pd.DataFrame(table2_rows)
    t1_df.to_csv(table1_csv, index=False)
    t2_df.to_csv(table2_csv, index=False)

    md = []
    md.append("Table 5.1 CatBoost E Protocol Metrics (Tuned Thresholds)\n")
    md.append(dataframe_to_markdown_plain(t1_df))
    md.append("\n")
    md.append("Note. MQTT FPR remains undefined because the test split contains no MQTT benign negatives.\n")
    md.append("\n")
    md.append("Table 5.2 CatBoost E Robustness Metrics (Tuned Thresholds)\n")
    md.append(dataframe_to_markdown_plain(t2_df))
    md.append("\n")
    md.append(
        "Note. In this surrogate FGSM/PGD export, only malicious rows are perturbed, so attacked benign FPR is undefined. "
        "Global selected threshold is model specific, not an arithmetic average.\n"
    )
    md.append("\n")
    table_md.write_text("\n".join(md), encoding="utf-8")

    manifest = {
        "generated_at_epoch_sec": time.time(),
        "elapsed_sec": time.time() - start_ts,
        "base_run_dir": str(base_run_dir),
        "matrix_run_dir": str(matrix_run_dir),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "protocols": [str(p) for p in protocols],
        "thresholds_used": thresholds,
        "attack_methods": [str(m) for m in attack_methods],
        "epsilons": [float(e) for e in epsilons],
        "train_sample_meta": train_sample_meta,
        "surrogate_meta": surrogate_meta,
        "rows_scanned": int(rows_scanned),
        "chunks_seen": int(chunks_seen),
        "files": {
            "table_5_1_csv": str(table1_csv),
            "table_5_2_csv": str(table2_csv),
            "tables_markdown": str(table_md),
            "manifest_json": str(manifest_json),
        },
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    log_progress(f"saved table 5.1 csv: {table1_csv}", start_ts=start_ts)
    log_progress(f"saved table 5.2 csv: {table2_csv}", start_ts=start_ts)
    log_progress(f"saved markdown tables: {table_md}", start_ts=start_ts)
    log_progress(f"saved manifest: {manifest_json}", start_ts=start_ts)


if __name__ == "__main__":
    main()
