                      

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def safe_float(value: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def stable_hash_u64(text: str) -> int:
    h = 1469598103934665603
    for ch in text:
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def reservoir_update(
    bucket: List[Tuple[List[float], int, str, str]],
    item: Tuple[List[float], int, str, str],
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(bucket) < capacity:
        bucket.append(item)
        return
    j = rng.randint(0, seen_count - 1)
    if j < capacity:
        bucket[j] = item


def read_header(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def load_train_val_samples(
    train_csv: str,
    feature_cols: Sequence[str],
    seed: int,
    train_cap_per_class: int,
    val_cap_per_class: int,
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    rng_train = {0: random.Random(seed + 101), 1: random.Random(seed + 202)}
    rng_val = {0: random.Random(seed + 303), 1: random.Random(seed + 404)}
    seen_train = defaultdict(int)
    seen_val = defaultdict(int)
    buckets_train: Dict[int, List[Tuple[List[float], int, str, str]]] = {0: [], 1: []}
    buckets_val: Dict[int, List[Tuple[List[float], int, str, str]]] = {0: [], 1: []}

    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row["label"])
            feats = [safe_float(row[c]) for c in feature_cols]
            protocol = row.get("protocol_hint", "unknown")
            family = row.get("attack_family", "unknown")
            key = f"{row.get('source_relpath','')}#{row.get('source_row_index','0')}#{seed}"
            h = stable_hash_u64(key)
            if h % 5 == 0:
                seen_val[y] += 1
                reservoir_update(
                    buckets_val[y],
                    (feats, y, protocol, family),
                    seen_val[y],
                    val_cap_per_class,
                    rng_val[y],
                )
            else:
                seen_train[y] += 1
                reservoir_update(
                    buckets_train[y],
                    (feats, y, protocol, family),
                    seen_train[y],
                    train_cap_per_class,
                    rng_train[y],
                )

    train_rows = buckets_train[0] + buckets_train[1]
    val_rows = buckets_val[0] + buckets_val[1]
    random.Random(seed + 505).shuffle(train_rows)
    random.Random(seed + 606).shuffle(val_rows)
    x_train = [r[0] for r in train_rows]
    y_train = [r[1] for r in train_rows]
    x_val = [r[0] for r in val_rows]
    y_val = [r[1] for r in val_rows]
    return x_train, y_train, x_val, y_val


def load_test_samples(
    test_csv: str,
    feature_cols: Sequence[str],
    seed: int,
    test_cap_per_class: int,
) -> Tuple[List[List[float]], List[int], List[str], List[str]]:
    rng_test = {0: random.Random(seed + 707), 1: random.Random(seed + 808)}
    seen = defaultdict(int)
    buckets: Dict[int, List[Tuple[List[float], int, str, str]]] = {0: [], 1: []}

    with open(test_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row["label"])
            feats = [safe_float(row[c]) for c in feature_cols]
            protocol = row.get("protocol_hint", "unknown")
            family = row.get("attack_family", "unknown")
            seen[y] += 1
            reservoir_update(
                buckets[y], (feats, y, protocol, family), seen[y], test_cap_per_class, rng_test[y]
            )

    rows = buckets[0] + buckets[1]
    random.Random(seed + 909).shuffle(rows)
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    protocol = [r[2] for r in rows]
    family = [r[3] for r in rows]
    return x, y, protocol, family


def compute_standardizer(x_train: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    d = len(x_train[0])
    means = [0.0] * d
    for row in x_train:
        for j, v in enumerate(row):
            means[j] += v
    n = float(len(x_train))
    means = [m / n for m in means]
    vars_ = [0.0] * d
    for row in x_train:
        for j, v in enumerate(row):
            dv = v - means[j]
            vars_[j] += dv * dv
    stds = [math.sqrt(v / max(1.0, n - 1.0)) for v in vars_]
    stds = [s if s > 1e-12 else 1.0 for s in stds]
    return means, stds


def apply_standardizer(x: Sequence[Sequence[float]], means: Sequence[float], stds: Sequence[float]) -> List[List[float]]:
    out = []
    for row in x:
        out.append([(v - means[j]) / stds[j] for j, v in enumerate(row)])
    return out


class LogisticRegressionSGD:
    def __init__(self, n_features: int, seed: int = 42, lr: float = 0.02, l2: float = 1e-4, epochs: int = 6):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.seed = seed
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        n = len(y)
        pos = sum(y)
        neg = n - pos
        cw = {0: n / (2.0 * max(1, neg)), 1: n / (2.0 * max(1, pos))}
        idx = list(range(n))
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:
                xi = x[i]
                yi = y[i]
                z = dot(self.w, xi) + self.b
                p = sigmoid(z)
                g = (p - yi) * cw[yi]
                for j, xv in enumerate(xi):
                    self.w[j] -= self.lr * (g * xv + self.l2 * self.w[j])
                self.b -= self.lr * g

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        return [sigmoid(dot(self.w, xi) + self.b) for xi in x]


class MeanThresholdStump:
    def __init__(self) -> None:
        self.feature_index = 0
        self.threshold = 0.0
        self.polarity = 1

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int], sample_weight: Sequence[float] | None = None) -> float:
        n = len(y)
        d = len(x[0])
        w = sample_weight if sample_weight is not None else [1.0 / n] * n
        best_err = float("inf")
        best_j = 0
        best_t = 0.0
        best_pol = 1
        for j in range(d):
            w0 = w1 = s0 = s1 = 0.0
            for i in range(n):
                if y[i] == 1:
                    w1 += w[i]
                    s1 += w[i] * x[i][j]
                else:
                    w0 += w[i]
                    s0 += w[i] * x[i][j]
            if w0 <= 0.0 or w1 <= 0.0:
                continue
            m0 = s0 / w0
            m1 = s1 / w1
            t = 0.5 * (m0 + m1)
            err_pol1 = 0.0
            err_polm1 = 0.0
            for i in range(n):
                pred1 = 1 if x[i][j] >= t else 0
                predm1 = 1 - pred1
                if pred1 != y[i]:
                    err_pol1 += w[i]
                if predm1 != y[i]:
                    err_polm1 += w[i]
            if err_pol1 < best_err:
                best_err = err_pol1
                best_j = j
                best_t = t
                best_pol = 1
            if err_polm1 < best_err:
                best_err = err_polm1
                best_j = j
                best_t = t
                best_pol = -1
        self.feature_index = best_j
        self.threshold = best_t
        self.polarity = best_pol
        return best_err

    def predict(self, x: Sequence[Sequence[float]]) -> List[int]:
        out = []
        j = self.feature_index
        t = self.threshold
        for row in x:
            p = 1 if row[j] >= t else 0
            if self.polarity < 0:
                p = 1 - p
            out.append(p)
        return out

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        return [float(p) for p in self.predict(x)]


class AdaBoostStumps:
    def __init__(self, rounds: int = 20):
        self.rounds = rounds
        self.stumps: List[MeanThresholdStump] = []
        self.alphas: List[float] = []

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        n = len(y)
        w = [1.0 / n] * n
        self.stumps = []
        self.alphas = []
        y_signed = [1 if yy == 1 else -1 for yy in y]
        for _ in range(self.rounds):
            stump = MeanThresholdStump()
            err = stump.fit(x, y, sample_weight=w)
            err = min(max(err, 1e-8), 0.499999)
            alpha = 0.5 * math.log((1.0 - err) / err)
            preds = stump.predict(x)
            for i in range(n):
                pred_signed = 1 if preds[i] == 1 else -1
                w[i] *= math.exp(-alpha * y_signed[i] * pred_signed)
            z = sum(w)
            if z <= 0:
                break
            w = [wi / z for wi in w]
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        out = []
        for row in x:
            s = 0.0
            for stump, alpha in zip(self.stumps, self.alphas):
                p = stump.predict([row])[0]
                s += alpha * (1.0 if p == 1 else -1.0)
            out.append(sigmoid(2.0 * s))
        return out


class SimpleMLP:
    def __init__(self, n_features: int, hidden: int = 16, seed: int = 42, lr: float = 0.01, epochs: int = 4):
        self.n_features = n_features
        self.hidden = hidden
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.05, 0.05) for _ in range(n_features)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [rng.uniform(-0.05, 0.05) for _ in range(hidden)]
        self.b2 = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        n = len(y)
        pos = sum(y)
        neg = n - pos
        cw = {0: n / (2.0 * max(1, neg)), 1: n / (2.0 * max(1, pos))}
        idx = list(range(n))
        rng = random.Random(self.seed + 17)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for i in idx:
                xi = x[i]
                yi = y[i]
                h = [0.0] * self.hidden
                for k in range(self.hidden):
                    z = dot(self.w1[k], xi) + self.b1[k]
                    h[k] = z if z > 0.0 else 0.0
                o = sigmoid(dot(self.w2, h) + self.b2)
                d2 = (o - yi) * cw[yi]
                for k in range(self.hidden):
                    self.w2[k] -= self.lr * d2 * h[k]
                self.b2 -= self.lr * d2
                for k in range(self.hidden):
                    if h[k] <= 0.0:
                        continue
                    d1 = d2 * self.w2[k]
                    for j in range(self.n_features):
                        self.w1[k][j] -= self.lr * d1 * xi[j]
                    self.b1[k] -= self.lr * d1

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        out = []
        for xi in x:
            h = [0.0] * self.hidden
            for k in range(self.hidden):
                z = dot(self.w1[k], xi) + self.b1[k]
                h[k] = z if z > 0.0 else 0.0
            out.append(sigmoid(dot(self.w2, h) + self.b2))
        return out


def threshold_for_max_recall_at_fpr(y_true: Sequence[int], p: Sequence[float], max_fpr: float = 0.01) -> Tuple[float, float, float]:
    pairs = sorted(zip(p, y_true), key=lambda t: t[0], reverse=True)
    p_total = sum(y_true)
    n_total = len(y_true) - p_total
    if p_total == 0 or n_total == 0:
        return 0.5, 0.0, 0.0
    tp = fp = 0
    best_threshold = 1.0
    best_recall = 0.0
    best_fpr = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / p_total
        fpr = fp / n_total
        if fpr <= max_fpr and recall >= best_recall:
            best_recall = recall
            best_threshold = score
            best_fpr = fpr
    return best_threshold, best_recall, best_fpr


def confusion_metrics(y_true: Sequence[int], p: Sequence[float], threshold: float) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    for yt, score in zip(y_true, p):
        yp = 1 if score >= threshold else 0
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)
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


def roc_auc(y_true: Sequence[int], p: Sequence[float]) -> float:
    pairs = sorted(zip(p, y_true))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_block = sum(y for _, y in pairs[i:j])
        rank_sum += avg_rank * pos_in_block
        i = j
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def average_precision(y_true: Sequence[int], p: Sequence[float]) -> float:
    pairs = sorted(zip(p, y_true), reverse=True)
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    tp = fp = 0
    ap_sum = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
            ap_sum += tp / max(1, tp + fp)
        else:
            fp += 1
    return ap_sum / n_pos


def evaluate_model(y_true: Sequence[int], p: Sequence[float], threshold: float) -> Dict[str, float]:
    cm = confusion_metrics(y_true, p, threshold)
    cm["roc_auc"] = roc_auc(y_true, p)
    cm["pr_auc"] = average_precision(y_true, p)
    return cm


def slice_metrics(
    y_true: Sequence[int],
    p: Sequence[float],
    threshold: float,
    slice_values: Sequence[str],
    slice_name: str,
) -> List[Dict[str, float | str]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(slice_values):
        groups[s].append(i)
    rows = []
    for s, idx in sorted(groups.items(), key=lambda t: (t[0] is None, str(t[0]))):
        ys = [y_true[i] for i in idx]
        ps = [p[i] for i in idx]
        m = evaluate_model(ys, ps, threshold)
        m[slice_name] = s
        m["n"] = len(idx)
        rows.append(m)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline IDS models with stdlib only")
    parser.add_argument("--train-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_train.csv")
    parser.add_argument("--test-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_test.csv")
    parser.add_argument("--output-dir", default="/home/capstone15/reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap-per-class", type=int, default=15000)
    parser.add_argument("--val-cap-per-class", type=int, default=5000)
    parser.add_argument("--test-cap-per-class", type=int, default=20000)
    args = parser.parse_args()

    header = read_header(args.train_csv)
    feature_cols = header[20:]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"baseline_models_stdlib_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val = load_train_val_samples(
        args.train_csv,
        feature_cols,
        seed=args.seed,
        train_cap_per_class=args.train_cap_per_class,
        val_cap_per_class=args.val_cap_per_class,
    )
    x_test, y_test, test_protocol, test_family = load_test_samples(
        args.test_csv, feature_cols, seed=args.seed, test_cap_per_class=args.test_cap_per_class
    )

    means, stds = compute_standardizer(x_train)
    x_train_s = apply_standardizer(x_train, means, stds)
    x_val_s = apply_standardizer(x_val, means, stds)
    x_test_s = apply_standardizer(x_test, means, stds)

    models = {
        "logistic_sgd": LogisticRegressionSGD(n_features=len(feature_cols), seed=args.seed),
        "tree_stump": MeanThresholdStump(),
        "adaboost_stumps": AdaBoostStumps(rounds=20),
        "mlp_1hidden": SimpleMLP(n_features=len(feature_cols), hidden=16, seed=args.seed, epochs=4),
    }

    full_metrics = {}
    summary_rows = []
    all_protocol_rows = []
    all_family_rows = []

    for name, model in models.items():
        if name in {"logistic_sgd", "mlp_1hidden"}:
            model.fit(x_train_s, y_train)
            p_val = model.predict_proba(x_val_s)
            p_test = model.predict_proba(x_test_s)
        else:
            model.fit(x_train_s, y_train)
            p_val = model.predict_proba(x_val_s)
            p_test = model.predict_proba(x_test_s)

        threshold, val_recall_at_fpr, val_fpr = threshold_for_max_recall_at_fpr(y_val, p_val, max_fpr=0.01)
        test_metrics = evaluate_model(y_test, p_test, threshold)
        test_metrics["threshold"] = threshold
        test_metrics["val_recall_at_fpr_le_1pct"] = val_recall_at_fpr
        test_metrics["val_fpr_at_selected_threshold"] = val_fpr
        test_metrics["model"] = name

        protocol_rows = slice_metrics(y_test, p_test, threshold, test_protocol, "protocol_hint")
        family_rows = slice_metrics(y_test, p_test, threshold, test_family, "attack_family")
        for r in protocol_rows:
            r["model"] = name
        for r in family_rows:
            r["model"] = name
        all_protocol_rows.extend(protocol_rows)
        all_family_rows.extend(family_rows)

        full_metrics[name] = {
            "global_test": test_metrics,
            "protocol_slices": protocol_rows,
            "attack_family_slices": family_rows,
        }
        summary_rows.append(
            {
                "model": name,
                "threshold": threshold,
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "fpr": test_metrics["fpr"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "tp": test_metrics["tp"],
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "val_recall_at_fpr_le_1pct": val_recall_at_fpr,
                "val_fpr_at_selected_threshold": val_fpr,
            }
        )

    summary_rows.sort(key=lambda r: r["f1"], reverse=True)
    best_model = summary_rows[0]["model"] if summary_rows else None

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "seed": args.seed,
                "feature_columns": feature_cols,
                "sample_sizes": {
                    "train": len(y_train),
                    "val": len(y_val),
                    "test": len(y_test),
                    "train_pos_ratio": (sum(y_train) / max(1, len(y_train))),
                    "val_pos_ratio": (sum(y_val) / max(1, len(y_val))),
                    "test_pos_ratio": (sum(y_test) / max(1, len(y_test))),
                },
                "best_model_by_f1": best_model,
                "results": full_metrics,
            },
            f,
            indent=2,
        )

    write_csv(
        str(out_dir / "metrics_summary.csv"),
        summary_rows,
        [
            "model",
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
            "val_recall_at_fpr_le_1pct",
            "val_fpr_at_selected_threshold",
        ],
    )
    write_csv(
        str(out_dir / "slice_metrics_protocol.csv"),
        all_protocol_rows,
        ["model", "protocol_hint", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )
    write_csv(
        str(out_dir / "slice_metrics_attack_family.csv"),
        all_family_rows,
        ["model", "attack_family", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )

    with open(out_dir / "RUN_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("Baseline training (stdlib) completed\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Best model by F1: {best_model}\n")
        f.write(f"Rows: train={len(y_train)} val={len(y_val)} test={len(y_test)}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")

    print(str(out_dir))


if __name__ == "__main__":
    main()
