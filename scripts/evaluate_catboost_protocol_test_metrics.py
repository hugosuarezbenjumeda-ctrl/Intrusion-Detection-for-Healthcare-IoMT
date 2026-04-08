                      

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DEFAULT_PROTOCOLS = ("wifi", "mqtt", "bluetooth")


def protocol_slug(protocol: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(protocol).strip().lower())
    slug = slug.strip("_")
    return slug if slug else "unknown"


def normalize_label_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)


def load_feature_columns(base_run_dir: Path) -> List[str]:
    metrics_path = base_run_dir / "metrics_summary.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cols = payload.get("feature_columns", [])
    if not isinstance(cols, list) or not cols:
        raise RuntimeError(f"Could not read feature columns from {metrics_path}")
    return [str(col) for col in cols]


def parse_protocols(raw: str) -> List[str]:
    parts = [protocol_slug(part) for part in str(raw).split(",") if str(part).strip()]
    unique = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    if not unique:
        raise ValueError("No protocols provided.")
    return unique


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return math.nan
    return float(num) / float(den)


def roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    pos = int(y.sum())
    neg = int(y.size - pos)
    if pos <= 0 or neg <= 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")[::-1]
    y_sorted = y[order]
    s_sorted = s[order]
    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted, dtype=np.int64)[threshold_idxs].astype(np.float64, copy=False)
    fps = (threshold_idxs + 1).astype(np.float64, copy=False) - tps
    tpr = np.r_[0.0, tps / float(pos)]
    fpr = np.r_[0.0, fps / float(neg)]
    return float(np.trapz(tpr, fpr))


def precision_recall_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    pos = int(y.sum())
    if pos <= 0:
        return np.array([1.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    order = np.argsort(s, kind="mergesort")[::-1]
    y_sorted = y[order]
    s_sorted = s[order]
    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted, dtype=np.int64)[threshold_idxs].astype(np.float64, copy=False)
    fps = (threshold_idxs + 1).astype(np.float64, copy=False) - tps
    precision = np.divide(tps, np.maximum(tps + fps, 1.0))
    recall = tps / float(pos)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return precision, recall


def average_precision_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall = precision_recall_points(y_true, scores)
    deltas = recall[1:] - recall[:-1]
    return float(np.sum(deltas * precision[1:]))


def pr_auc_trapezoid_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall = precision_recall_points(y_true, scores)
    return float(np.trapz(precision, recall))


def log_loss_manual(y_true: np.ndarray, scores: np.ndarray, eps: float = 1e-15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    p = np.clip(s, eps, 1.0 - eps)
    return float(-np.mean((y * np.log(p)) + ((1.0 - y) * np.log(1.0 - p))))


def brier_score_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    return float(np.mean((s - y) ** 2))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y = np.asarray(y_true, dtype=np.uint8)
    pred = np.asarray(y_pred, dtype=np.uint8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, tn, fp, fn


def summarize_scope(scope: str, threshold_repr: Any, y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    n_rows = int(len(y_true))
    support_attack = int(tp + fn)
    support_benign = int(tn + fp)
    predicted_attack = int(tp + fp)
    predicted_benign = int(tn + fn)

    accuracy = safe_div(tp + tn, n_rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, (2 * tp) + fp + fn)
    f05 = safe_div((1.0 + 0.5 ** 2) * precision * recall, (0.5 ** 2 * precision) + recall)
    f2 = safe_div((1.0 + 2.0 ** 2) * precision * recall, (2.0 ** 2 * precision) + recall)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    npv = safe_div(tn, tn + fn)
    fdr = safe_div(fp, fp + tp)
    false_omission_rate = safe_div(fn, fn + tn)
    prevalence = safe_div(support_attack, n_rows)
    predicted_positive_rate = safe_div(predicted_attack, n_rows)
    predicted_negative_rate = safe_div(predicted_benign, n_rows)
    balanced_accuracy = (
        math.nan
        if math.isnan(recall) or math.isnan(specificity)
        else float((recall + specificity) / 2.0)
    )
    gmean = (
        math.nan
        if math.isnan(recall) or math.isnan(specificity) or recall < 0.0 or specificity < 0.0
        else float(math.sqrt(recall * specificity))
    )
    jaccard = safe_div(tp, tp + fp + fn)
    youdens_j = (
        math.nan
        if math.isnan(recall) or math.isnan(specificity)
        else float(recall + specificity - 1.0)
    )
    markedness = (
        math.nan
        if math.isnan(precision) or math.isnan(npv)
        else float(precision + npv - 1.0)
    )
    mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = math.nan if mcc_den == 0.0 else float(((tp * tn) - (fp * fn)) / mcc_den)
    expected_accuracy = (
        (safe_div(support_attack, n_rows) * safe_div(predicted_attack, n_rows))
        + (safe_div(support_benign, n_rows) * safe_div(predicted_benign, n_rows))
    )
    kappa = math.nan if expected_accuracy == 1.0 else safe_div(accuracy - expected_accuracy, 1.0 - expected_accuracy)
    lr_positive = math.nan if math.isnan(recall) or math.isnan(fpr) or fpr == 0.0 else float(recall / fpr)
    lr_negative = math.nan if math.isnan(fnr) or math.isnan(specificity) or specificity == 0.0 else float(fnr / specificity)
    dor = math.nan if math.isnan(lr_positive) or math.isnan(lr_negative) or lr_negative == 0.0 else float(lr_positive / lr_negative)

    score_mean = float(np.mean(scores)) if n_rows else math.nan
    score_std = float(np.std(scores)) if n_rows else math.nan
    score_min = float(np.min(scores)) if n_rows else math.nan
    score_max = float(np.max(scores)) if n_rows else math.nan
    score_p01 = float(np.quantile(scores, 0.01)) if n_rows else math.nan
    score_p05 = float(np.quantile(scores, 0.05)) if n_rows else math.nan
    score_p50 = float(np.quantile(scores, 0.50)) if n_rows else math.nan
    score_p95 = float(np.quantile(scores, 0.95)) if n_rows else math.nan
    score_p99 = float(np.quantile(scores, 0.99)) if n_rows else math.nan
    attack_score_mean = float(np.mean(scores[y_true == 1])) if support_attack else math.nan
    benign_score_mean = float(np.mean(scores[y_true == 0])) if support_benign else math.nan

    roc_auc = roc_auc_manual(y_true, scores)
    pr_auc_step = average_precision_manual(y_true, scores)
    pr_auc_trapz = pr_auc_trapezoid_manual(y_true, scores)
    log_loss = log_loss_manual(y_true, scores)
    brier_score = brier_score_manual(y_true, scores)

    return {
        "scope": scope,
        "threshold": threshold_repr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f0_5": f05,
        "f2": f2,
        "jaccard": jaccard,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "gmean": gmean,
        "youden_j": youdens_j,
        "markedness": markedness,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": recall,
        "tnr": specificity,
        "npv": npv,
        "fdr": fdr,
        "false_omission_rate": false_omission_rate,
        "lr_positive": lr_positive,
        "lr_negative": lr_negative,
        "diagnostic_odds_ratio": dor,
        "roc_auc": roc_auc,
        "pr_auc_step": pr_auc_step,
        "pr_auc_trapz": pr_auc_trapz,
        "log_loss": log_loss,
        "brier_score": brier_score,
        "prevalence": prevalence,
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": predicted_negative_rate,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_min": score_min,
        "score_max": score_max,
        "score_p01": score_p01,
        "score_p05": score_p05,
        "score_p50": score_p50,
        "score_p95": score_p95,
        "score_p99": score_p99,
        "attack_score_mean": attack_score_mean,
        "benign_score_mean": benign_score_mean,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_attack": support_attack,
        "support_benign": support_benign,
        "predicted_attack": predicted_attack,
        "predicted_benign": predicted_benign,
        "n_rows": n_rows,
    }


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if not math.isfinite(value_f):
            return None
        return value_f
    return value


def load_protocol_bundle(
    matrix_run_dir: Path,
    protocol: str,
    model_name: str,
    family_id: str,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    candidate_dir = (
        matrix_run_dir
        / "candidates"
        / f"{stage_name}_{protocol}_{model_name}_{family_id}_seed_{int(seed)}"
    )
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

    threshold = float(summary["selected_threshold"])
    return {
        "protocol": protocol,
        "threshold": threshold,
        "summary_path": summary_path,
        "model_path": model_path,
        "candidate_summary": summary,
        "model": model,
    }


def predict_chunk(
    chunk: pd.DataFrame,
    feature_columns: List[str],
    bundles_by_protocol: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, np.ndarray]]:
    protocol_values = chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
    y_true = normalize_label_series(chunk["label"])
    outputs: Dict[str, Dict[str, np.ndarray]] = {}

    for protocol in bundles_by_protocol:
        mask = protocol_values == protocol
        if not np.any(mask):
            continue
        x_df = chunk.loc[mask, feature_columns].copy()
        for col in feature_columns:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce").fillna(0.0)
        x_mat = x_df.to_numpy(dtype=np.float32, copy=False)
        bundle = bundles_by_protocol[protocol]
        score = bundle["model"].predict_proba(x_mat)[:, 1].astype(np.float32, copy=False)
        threshold = float(bundle["threshold"])
        pred = (score >= threshold).astype(np.uint8, copy=False)
        outputs[protocol] = {
            "y_true": y_true[mask].astype(np.uint8, copy=False),
            "score": score,
            "y_pred": pred,
        }
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851"),
        help="Base full-data HPO run with metrics_summary.json feature columns.",
    )
    ap.add_argument(
        "--matrix-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105"),
        help="Robust matrix run containing candidate summaries and saved CatBoost models.",
    )
    ap.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/merged/metadata_test.csv"),
        help="Merged metadata test CSV.",
    )
    ap.add_argument("--protocols", type=str, default="wifi,mqtt,bluetooth")
    ap.add_argument("--model-name", type=str, default="catboost")
    ap.add_argument("--family-id", type=str, default="E")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage-name", type=str, default="coarse")
    ap.add_argument("--chunk-size", type=int, default=200000)
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write the extended metrics CSV.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the evaluation manifest JSON.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    start_ts = time.time()

    protocols = parse_protocols(args.protocols)
    feature_columns = load_feature_columns(args.base_run_dir)
    bundles_by_protocol = {
        protocol: load_protocol_bundle(
            matrix_run_dir=args.matrix_run_dir,
            protocol=protocol,
            model_name=args.model_name,
            family_id=args.family_id,
            seed=args.seed,
            stage_name=args.stage_name,
        )
        for protocol in protocols
    }

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else args.matrix_run_dir / f"{args.model_name}_{args.family_id}_test_predictions_metrics_extended.csv"
    )
    output_json = (
        args.output_json
        if args.output_json is not None
        else args.matrix_run_dir / f"{args.model_name}_{args.family_id}_test_predictions_metrics_extended_manifest.json"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    usecols = ["protocol_hint", "label"] + feature_columns
    dtypes = {col: "float32" for col in feature_columns}
    dtypes.update({"protocol_hint": "string", "label": "int8"})

    labels_by_scope: Dict[str, List[np.ndarray]] = {scope: [] for scope in protocols}
    labels_by_scope["global_protocol_routed"] = []
    scores_by_scope: Dict[str, List[np.ndarray]] = {scope: [] for scope in protocols}
    scores_by_scope["global_protocol_routed"] = []
    preds_by_scope: Dict[str, List[np.ndarray]] = {scope: [] for scope in protocols}
    preds_by_scope["global_protocol_routed"] = []

    rows_scanned = 0
    chunks_seen = 0
    reader = pd.read_csv(
        args.test_csv,
        usecols=usecols,
        dtype=dtypes,
        chunksize=max(1, int(args.chunk_size)),
    )
    for chunk in reader:
        chunks_seen += 1
        rows_scanned += int(len(chunk))
        chunk_outputs = predict_chunk(chunk=chunk, feature_columns=feature_columns, bundles_by_protocol=bundles_by_protocol)
        for protocol, payload in chunk_outputs.items():
            labels_by_scope[protocol].append(payload["y_true"])
            scores_by_scope[protocol].append(payload["score"])
            preds_by_scope[protocol].append(payload["y_pred"])
            labels_by_scope["global_protocol_routed"].append(payload["y_true"])
            scores_by_scope["global_protocol_routed"].append(payload["score"])
            preds_by_scope["global_protocol_routed"].append(payload["y_pred"])

        elapsed = time.time() - start_ts
        print(f"[{elapsed:8.1f}s] processed chunk={chunks_seen} rows_scanned={rows_scanned}")

    rows: List[Dict[str, Any]] = []
    for protocol in protocols:
        y_true = np.concatenate(labels_by_scope[protocol], axis=0) if labels_by_scope[protocol] else np.empty(0, dtype=np.uint8)
        scores = np.concatenate(scores_by_scope[protocol], axis=0) if scores_by_scope[protocol] else np.empty(0, dtype=np.float32)
        y_pred = np.concatenate(preds_by_scope[protocol], axis=0) if preds_by_scope[protocol] else np.empty(0, dtype=np.uint8)
        rows.append(
            summarize_scope(
                scope=protocol,
                threshold_repr=float(bundles_by_protocol[protocol]["threshold"]),
                y_true=y_true,
                scores=scores,
                y_pred=y_pred,
            )
        )

    global_y_true = np.concatenate(labels_by_scope["global_protocol_routed"], axis=0)
    global_scores = np.concatenate(scores_by_scope["global_protocol_routed"], axis=0)
    global_y_pred = np.concatenate(preds_by_scope["global_protocol_routed"], axis=0)
    rows.append(
        summarize_scope(
            scope="global_protocol_routed",
            threshold_repr="protocol_specific",
            y_true=global_y_true,
            scores=global_scores,
            y_pred=global_y_pred,
        )
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_csv, index=False)

    manifest = {
        "generated_at_epoch_sec": time.time(),
        "elapsed_sec": time.time() - start_ts,
        "base_run_dir": args.base_run_dir,
        "matrix_run_dir": args.matrix_run_dir,
        "test_csv": args.test_csv,
        "protocols": protocols,
        "model_name": args.model_name,
        "family_id": args.family_id,
        "seed": int(args.seed),
        "stage_name": args.stage_name,
        "chunk_size": int(args.chunk_size),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "rows_scanned": int(rows_scanned),
        "chunks_seen": int(chunks_seen),
        "output_csv": output_csv,
        "protocol_artifacts": {
            protocol: {
                "threshold": bundles_by_protocol[protocol]["threshold"],
                "summary_path": bundles_by_protocol[protocol]["summary_path"],
                "model_path": bundles_by_protocol[protocol]["model_path"],
            }
            for protocol in protocols
        },
        "notes": [
            "MQTT test rows contain only positive labels, so metrics requiring benign negatives remain undefined there.",
            "Thresholds are the saved protocol-specific selected_threshold values from the stable catboost family E artifacts.",
        ],
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(sanitize_json_value(manifest), f, indent=2)

    print(f"Wrote metrics CSV: {output_csv}")
    print(f"Wrote manifest JSON: {output_json}")


if __name__ == "__main__":
    main()
