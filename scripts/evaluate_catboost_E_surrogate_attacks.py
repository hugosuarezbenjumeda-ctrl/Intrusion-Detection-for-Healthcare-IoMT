                      

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

from evaluate_xgb_robustness import (
    attack_surrogate_fgsm,
    attack_surrogate_pgd,
    build_constraints,
    classification_metrics,
    compute_asr,
    log_progress,
    parse_epsilons,
    sample_train_rows_by_protocol,
    to_jsonable,
    train_surrogates_by_protocol,
)
from xgb_protocol_ids_utils import load_feature_columns, prepare_feature_matrix, protocol_slug


DEFAULT_PROTOCOLS = ("wifi", "mqtt", "bluetooth")
ALLOWED_ATTACK_METHODS = ("surrogate_fgsm", "surrogate_pgd")


@dataclass
class MetricAccumulator:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    n: int = 0
    n_benign: int = 0
    n_attack: int = 0
    asr_conditional_num: int = 0
    asr_conditional_den: int = 0
    asr_all_num: int = 0
    asr_all_den: int = 0

    def update(self, y_true: np.ndarray, baseline_pred: np.ndarray, adv_pred: np.ndarray) -> None:
        cls = classification_metrics(y_true=y_true, y_pred=adv_pred)
        self.tp += int(cls["tp"])
        self.tn += int(cls["tn"])
        self.fp += int(cls["fp"])
        self.fn += int(cls["fn"])
        self.n += int(cls["n"])
        self.n_benign += int(cls["n_benign"])
        self.n_attack += int(cls["n_attack"])

        scope_mask = np.ones(len(y_true), dtype=bool)
        asr = compute_asr(
            y_true=y_true,
            baseline_pred=baseline_pred,
            adv_pred=adv_pred,
            scope_mask=scope_mask,
        )
        self.asr_conditional_num += int(asr["asr_conditional_num"])
        self.asr_conditional_den += int(asr["asr_conditional_den"])
        self.asr_all_num += int(asr["asr_all_num"])
        self.asr_all_den += int(asr["asr_all_den"])


def parse_protocols(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).split(","):
        proto = protocol_slug(token)
        if not proto:
            continue
        if proto in out:
            continue
        out.append(proto)
    if not out:
        raise ValueError("No protocols provided.")
    return out


def parse_attack_methods(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).split(","):
        method = str(token).strip()
        if not method:
            continue
        if method not in ALLOWED_ATTACK_METHODS:
            raise ValueError(
                f"Unknown attack method '{method}'. Valid methods: {','.join(ALLOWED_ATTACK_METHODS)}"
            )
        if method in out:
            continue
        out.append(method)
    if not out:
        raise ValueError("No attack methods provided.")
    return out


def choose_default_protocol(protocols: Sequence[str]) -> str:
    for proto in protocols:
        if proto == "wifi":
            return "wifi"
    return str(protocols[0])


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return float("nan")
    return float(num) / float(den)


def metrics_from_acc(acc: MetricAccumulator) -> Dict[str, Any]:
    precision = safe_div(acc.tp, acc.tp + acc.fp)
    recall = safe_div(acc.tp, acc.tp + acc.fn)
    if math.isnan(precision) or math.isnan(recall) or float(precision + recall) == 0.0:
        f1 = 0.0
    else:
        f1 = float((2.0 * precision * recall) / (precision + recall))
    fpr = safe_div(acc.fp, acc.fp + acc.tn)

    asr_conditional = safe_div(acc.asr_conditional_num, acc.asr_conditional_den)
    asr_all = safe_div(acc.asr_all_num, acc.asr_all_den)

    return {
        "tp": int(acc.tp),
        "tn": int(acc.tn),
        "fp": int(acc.fp),
        "fn": int(acc.fn),
        "precision": float(precision) if not math.isnan(precision) else float("nan"),
        "recall": float(recall) if not math.isnan(recall) else float("nan"),
        "f1": float(f1),
        "fpr": float(fpr) if not math.isnan(fpr) else float("nan"),
        "n": int(acc.n),
        "n_benign": int(acc.n_benign),
        "n_attack": int(acc.n_attack),
        "asr_conditional": float(asr_conditional) if not math.isnan(asr_conditional) else float("nan"),
        "asr_conditional_num": int(acc.asr_conditional_num),
        "asr_conditional_den": int(acc.asr_conditional_den),
        "asr_all": float(asr_all) if not math.isnan(asr_all) else float("nan"),
        "asr_all_num": int(acc.asr_all_num),
        "asr_all_den": int(acc.asr_all_den),
    }


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
        raise ValueError(f"Unsupported stage-name '{stage_name}'. Use coarse or stability.")

    summary_path = candidate_dir / "candidate_summary.json"
    model_path = candidate_dir / "models" / f"{protocol}__{model_name}__{family_id}.cbm"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing candidate summary: {summary_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    threshold = float(summary["selected_threshold"])
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    return {
        "protocol": protocol,
        "threshold": threshold,
        "candidate_dir": candidate_dir,
        "summary_path": summary_path,
        "model_path": model_path,
        "summary": summary,
        "model": model,
    }


def ensure_accumulators(
    attack_methods: Sequence[str],
    epsilons: Sequence[float],
    protocols: Sequence[str],
) -> Tuple[Dict[str, MetricAccumulator], Dict[Tuple[str, float, str], MetricAccumulator]]:
    baseline: Dict[str, MetricAccumulator] = {"global": MetricAccumulator()}
    for proto in protocols:
        baseline[str(proto)] = MetricAccumulator()

    attacks: Dict[Tuple[str, float, str], MetricAccumulator] = {}
    for method in attack_methods:
        for eps in epsilons:
            eps_f = float(eps)
            attacks[(str(method), eps_f, "global")] = MetricAccumulator()
            for proto in protocols:
                attacks[(str(method), eps_f, str(proto))] = MetricAccumulator()
    return baseline, attacks


def attack_malicious_subset(
    *,
    method: str,
    epsilon: float,
    x_malicious: np.ndarray,
    surrogate: Dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    locked_idx: np.ndarray,
    pgd_steps: int,
    pgd_alpha_ratio: float,
) -> np.ndarray:
    eps = float(epsilon)
    if eps <= 0.0 or x_malicious.size == 0:
        return x_malicious.copy()

    if method == "surrogate_fgsm":
        return attack_surrogate_fgsm(
            x_orig=x_malicious,
            surrogate=surrogate,
            lower=lower,
            upper=upper,
            locked_idx=locked_idx,
            epsilon=eps,
        )
    if method == "surrogate_pgd":
        return attack_surrogate_pgd(
            x_orig=x_malicious,
            surrogate=surrogate,
            lower=lower,
            upper=upper,
            locked_idx=locked_idx,
            epsilon=eps,
            steps=int(pgd_steps),
            alpha_ratio=float(pgd_alpha_ratio),
        )

    raise RuntimeError(f"Unsupported attack method '{method}'.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851"),
        help="Base HPO run directory containing metrics_summary.json with feature columns.",
    )
    ap.add_argument(
        "--matrix-run-dir",
        type=Path,
        default=Path("reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105"),
        help="Robust-matrix run directory containing saved CatBoost E artifacts.",
    )
    ap.add_argument("--train-csv", type=Path, default=Path("data/merged/metadata_train.csv"))
    ap.add_argument("--test-csv", type=Path, default=Path("data/merged/metadata_test.csv"))
    ap.add_argument("--protocols", type=str, default=",".join(DEFAULT_PROTOCOLS))
    ap.add_argument("--model-name", type=str, default="catboost")
    ap.add_argument("--family-id", type=str, default="E")
    ap.add_argument("--stage-name", type=str, choices=("coarse", "stability"), default="coarse")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epsilons", type=str, default="0,0.01,0.02,0.05,0.10")
    ap.add_argument("--attack-methods", type=str, default="surrogate_fgsm,surrogate_pgd")
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--pgd-alpha-ratio", type=float, default=0.25)

    ap.add_argument("--surrogate-train-per-protocol", type=int, default=200000)
    ap.add_argument("--surrogate-epochs", type=int, default=12)
    ap.add_argument("--surrogate-lr", type=float, default=0.08)
    ap.add_argument("--surrogate-batch-size", type=int, default=4096)
    ap.add_argument("--percentile-lower", type=float, default=1.0)
    ap.add_argument("--percentile-upper", type=float, default=99.0)

    ap.add_argument("--chunk-size", type=int, default=200000)
    ap.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Optional cap for quick local debugging. <=0 means full test CSV.",
    )

    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--constraints-csv", type=Path, default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    start_ts = time.time()

    protocols = parse_protocols(args.protocols)
    attack_methods = parse_attack_methods(args.attack_methods)
    epsilons = parse_epsilons(args.epsilons)

    fallback_protocol = choose_default_protocol(protocols)

    base_run_dir = args.base_run_dir.resolve()
    matrix_run_dir = args.matrix_run_dir.resolve()
    train_csv = args.train_csv.resolve()
    test_csv = args.test_csv.resolve()

    feature_columns = load_feature_columns(base_run_dir)

    log_progress("loading CatBoost artifacts", start_ts=start_ts)
    bundles: Dict[str, Dict[str, Any]] = {}
    for proto in protocols:
        bundles[proto] = load_catboost_bundle(
            matrix_run_dir=matrix_run_dir,
            protocol=proto,
            model_name=str(args.model_name),
            family_id=str(args.family_id),
            seed=int(args.seed),
            stage_name=str(args.stage_name),
        )

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else matrix_run_dir
        / f"{args.model_name}_{args.family_id}_surrogate_fgsm_pgd_metrics_{args.stage_name}_seed_{int(args.seed)}.csv"
    )
    output_json = (
        args.output_json
        if args.output_json is not None
        else matrix_run_dir
        / f"{args.model_name}_{args.family_id}_surrogate_fgsm_pgd_manifest_{args.stage_name}_seed_{int(args.seed)}.json"
    )
    constraints_csv = (
        args.constraints_csv
        if args.constraints_csv is not None
        else matrix_run_dir
        / f"{args.model_name}_{args.family_id}_surrogate_constraints_{args.stage_name}_seed_{int(args.seed)}.csv"
    )

    output_csv = output_csv.resolve()
    output_json = output_json.resolve()
    constraints_csv = constraints_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    constraints_csv.parent.mkdir(parents=True, exist_ok=True)

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

    log_progress("building realistic constraints", start_ts=start_ts)
    constraints, constraints_df = build_constraints(
        train_sample_df=train_sample_df,
        feature_columns=feature_columns,
        protocols=protocols,
        percentile_lower=float(args.percentile_lower),
        percentile_upper=float(args.percentile_upper),
    )
    constraints_df.to_csv(constraints_csv, index=False)

    log_progress("training protocol surrogates", start_ts=start_ts)
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

    baseline_acc, attack_acc = ensure_accumulators(
        attack_methods=attack_methods,
        epsilons=epsilons,
        protocols=protocols,
    )

    usecols = ["protocol_hint", "label"] + feature_columns
    chunk_size = max(1, int(args.chunk_size))

    rows_scanned = 0
    chunks_seen = 0
    unknown_protocol_rows = 0

    log_progress("running full test evaluation under attacks", start_ts=start_ts)
    reader = pd.read_csv(test_csv, usecols=usecols, chunksize=chunk_size)
    for chunk in reader:
        if int(args.max_test_rows) > 0 and rows_scanned >= int(args.max_test_rows):
            break

        if int(args.max_test_rows) > 0:
            remaining = int(args.max_test_rows) - int(rows_scanned)
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        chunks_seen += 1
        rows_scanned += int(len(chunk))

        protocol_hint_raw = chunk["protocol_hint"].fillna("").astype(str).map(protocol_slug).to_numpy(dtype=object, copy=False)
        routed_protocol = np.array(
            [p if p in bundles else fallback_protocol for p in protocol_hint_raw],
            dtype=object,
        )
        unknown_protocol_rows += int(np.sum(~np.isin(protocol_hint_raw, protocols)))

        y_chunk = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(np.int8).clip(0, 1).to_numpy(copy=False)
        x_chunk = prepare_feature_matrix(chunk.copy(), feature_columns)

        for proto in protocols:
            mask = routed_protocol == proto
            if not np.any(mask):
                continue

            idx = np.flatnonzero(mask)
            x_proto = x_chunk[idx]
            y_proto = y_chunk[idx]
            bundle = bundles[proto]
            threshold = float(bundle["threshold"])

            base_score = bundle["model"].predict_proba(x_proto)[:, 1].astype(np.float32, copy=False)
            base_pred = (base_score >= threshold).astype(np.int8, copy=False)

            baseline_acc[proto].update(y_true=y_proto, baseline_pred=base_pred, adv_pred=base_pred)
            baseline_acc["global"].update(y_true=y_proto, baseline_pred=base_pred, adv_pred=base_pred)

            malicious_mask = y_proto == 1
            x_mal = x_proto[malicious_mask] if np.any(malicious_mask) else np.empty((0, x_proto.shape[1]), dtype=np.float32)
            lower = constraints[proto]["lower"].astype(np.float64)
            upper = constraints[proto]["upper"].astype(np.float64)
            locked_idx = constraints[proto]["locked_idx"]

            for method in attack_methods:
                for eps in epsilons:
                    eps_f = float(eps)
                    if eps_f <= 0.0 or x_mal.size == 0:
                        adv_pred = base_pred
                    else:
                        x_adv_mal = attack_malicious_subset(
                            method=method,
                            epsilon=eps_f,
                            x_malicious=x_mal,
                            surrogate=surrogates[proto],
                            lower=lower,
                            upper=upper,
                            locked_idx=locked_idx,
                            pgd_steps=int(args.pgd_steps),
                            pgd_alpha_ratio=float(args.pgd_alpha_ratio),
                        )
                        adv_score_mal = bundle["model"].predict_proba(x_adv_mal)[:, 1].astype(np.float32, copy=False)
                        adv_pred = base_pred.copy()
                        adv_pred[malicious_mask] = (adv_score_mal >= threshold).astype(np.int8, copy=False)

                    attack_acc[(method, eps_f, proto)].update(
                        y_true=y_proto,
                        baseline_pred=base_pred,
                        adv_pred=adv_pred,
                    )
                    attack_acc[(method, eps_f, "global")].update(
                        y_true=y_proto,
                        baseline_pred=base_pred,
                        adv_pred=adv_pred,
                    )

        log_progress(
            f"processed chunks={chunks_seen} rows={rows_scanned}",
            start_ts=start_ts,
        )

    scope_order = ["global"] + list(protocols)
    baseline_f1: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for scope in scope_order:
        metrics = metrics_from_acc(baseline_acc[scope])
        baseline_f1[scope] = float(metrics["f1"])
        rows.append(
            {
                "attack_method": "baseline",
                "epsilon": 0.0,
                "scope": "global" if scope == "global" else "protocol",
                "protocol_hint": "" if scope == "global" else scope,
                **metrics,
                "delta_f1": 0.0,
            }
        )

    for method in attack_methods:
        for eps in epsilons:
            eps_f = float(eps)
            for scope in scope_order:
                metrics = metrics_from_acc(attack_acc[(method, eps_f, scope)])
                rows.append(
                    {
                        "attack_method": str(method),
                        "epsilon": eps_f,
                        "scope": "global" if scope == "global" else "protocol",
                        "protocol_hint": "" if scope == "global" else scope,
                        **metrics,
                        "delta_f1": float(metrics["f1"] - baseline_f1[scope]),
                    }
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(
        by=["attack_method", "epsilon", "scope", "protocol_hint"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    metrics_df.to_csv(output_csv, index=False)

    manifest = {
        "generated_at_epoch_sec": time.time(),
        "elapsed_sec": time.time() - start_ts,
        "base_run_dir": str(base_run_dir),
        "matrix_run_dir": str(matrix_run_dir),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "protocols": [str(p) for p in protocols],
        "default_protocol": str(fallback_protocol),
        "model_name": str(args.model_name),
        "family_id": str(args.family_id),
        "stage_name": str(args.stage_name),
        "seed": int(args.seed),
        "attack_methods": [str(m) for m in attack_methods],
        "epsilons": [float(e) for e in epsilons],
        "pgd_steps": int(args.pgd_steps),
        "pgd_alpha_ratio": float(args.pgd_alpha_ratio),
        "surrogate_train_per_protocol": int(args.surrogate_train_per_protocol),
        "surrogate_epochs": int(args.surrogate_epochs),
        "surrogate_lr": float(args.surrogate_lr),
        "surrogate_batch_size": int(args.surrogate_batch_size),
        "percentile_lower": float(args.percentile_lower),
        "percentile_upper": float(args.percentile_upper),
        "chunk_size": int(chunk_size),
        "max_test_rows": int(args.max_test_rows),
        "feature_count": int(len(feature_columns)),
        "rows_scanned": int(rows_scanned),
        "chunks_seen": int(chunks_seen),
        "unknown_protocol_rows_routed_to_default": int(unknown_protocol_rows),
        "train_sample_meta": train_sample_meta,
        "surrogate_meta": surrogate_meta,
        "protocol_artifacts": {
            proto: {
                "threshold": float(bundle["threshold"]),
                "candidate_dir": str(bundle["candidate_dir"]),
                "summary_path": str(bundle["summary_path"]),
                "model_path": str(bundle["model_path"]),
            }
            for proto, bundle in bundles.items()
        },
        "outputs": {
            "metrics_csv": str(output_csv),
            "manifest_json": str(output_json),
            "constraints_csv": str(constraints_csv),
        },
        "notes": [
            "Surrogate FGSM/PGD attacks are applied to malicious rows only; benign rows remain unchanged.",
            "Predictions use saved protocol-specific thresholds from CatBoost family-E candidate summaries.",
        ],
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    log_progress(f"saved metrics csv: {output_csv}", start_ts=start_ts)
    log_progress(f"saved manifest json: {output_json}", start_ts=start_ts)
    log_progress(f"saved constraints csv: {constraints_csv}", start_ts=start_ts)


if __name__ == "__main__":
    main()
