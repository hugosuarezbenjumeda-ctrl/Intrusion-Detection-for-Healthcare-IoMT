                      

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_feature_cols(train_csv: str) -> List[str]:
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return header[20:]


def build_val_mask(df: pd.DataFrame, seed: int) -> np.ndarray:
    keys = pd.util.hash_pandas_object(
        df[["source_relpath", "source_row_index"]].astype({"source_relpath": "string"}),
        index=False,
    ).to_numpy(np.uint64)
    return ((keys + np.uint64(seed)) % np.uint64(5)) == 0


def select_threshold_recall_at_fpr(y_true: np.ndarray, proba: np.ndarray, max_fpr: float = 0.01) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, proba)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx) == 0:
        return {"threshold": 1.0, "val_recall": 0.0, "val_fpr": 0.0}
    best = idx[np.argmax(tpr[idx])]
    return {"threshold": float(thr[best]), "val_recall": float(tpr[best]), "val_fpr": float(fpr[best])}


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def eval_by_slice(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    group_name: str,
    model_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for g in sorted(pd.Series(groups).astype(str).unique().tolist()):
        m = groups.astype(str) == g
        if int(m.sum()) == 0:
            continue
        out = evaluate(y_true[m], proba[m], threshold)
        out.update({"model": model_name, group_name: g, "n": int(m.sum())})
        rows.append(out)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class DeepMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class NNResult:
    val_proba: np.ndarray
    test_proba: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    best_val_aucpr: float
    state_dict: Dict[str, torch.Tensor]


def train_complex_nn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    device: str,
    epochs: int = 14,
    batch_size: int = 32768,
) -> NNResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std

    model = DeepMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_train = TensorDataset(
        torch.from_numpy(x_train_n.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    x_val_t = torch.from_numpy(x_val_n.astype(np.float32)).to(device)
    x_test_t = torch.from_numpy(x_test_n.astype(np.float32)).to(device)

    best_aucpr = -1.0
    best_state = None
    best_epoch = -1
    patience = 3
    stale = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_train:
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
        val_aucpr = float(average_precision_score(y_val, val_proba))

        if val_aucpr > best_aucpr:
            best_aucpr = val_aucpr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_t).detach().cpu().numpy()
        test_logits = model(x_test_t).detach().cpu().numpy()
    val_proba = 1.0 / (1.0 + np.exp(-val_logits))
    test_proba = 1.0 / (1.0 + np.exp(-test_logits))

    return NNResult(
        val_proba=val_proba,
        test_proba=test_proba,
        mean=mean,
        std=std,
        best_epoch=best_epoch,
        best_val_aucpr=best_aucpr,
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_train.csv")
    ap.add_argument("--test-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_test.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default="/home/capstone15/reports")
    args = ap.parse_args()

    np.random.seed(args.seed)

    feature_cols = load_feature_cols(args.train_csv)
    train_usecols = ["source_relpath", "source_row_index", "label"] + feature_cols
    test_usecols = ["label", "protocol_hint", "attack_family"] + feature_cols

    dtypes = {c: "float32" for c in feature_cols}
    dtypes.update({"label": "int8", "source_row_index": "int32"})

    train_df = pd.read_csv(args.train_csv, usecols=train_usecols, dtype=dtypes)
    val_mask = build_val_mask(train_df, args.seed)

    x_train = train_df.loc[~val_mask, feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df.loc[~val_mask, "label"].to_numpy(dtype=np.int8, copy=False)
    x_val = train_df.loc[val_mask, feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_val = train_df.loc[val_mask, "label"].to_numpy(dtype=np.int8, copy=False)
    del train_df

    test_df = pd.read_csv(args.test_csv, usecols=test_usecols, dtype={**dtypes, "protocol_hint": "string", "attack_family": "string"})
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df["label"].to_numpy(dtype=np.int8, copy=False)
    g_protocol = test_df["protocol_hint"].fillna("unknown").astype(str).to_numpy()
    g_family = test_df["attack_family"].fillna("unknown").astype(str).to_numpy()
    del test_df

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"full_gpu_models_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

                 
    xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric="aucpr",
    )
    xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
    p_val_xgb = xgb.predict_proba(x_val)[:, 1]
    p_test_xgb = xgb.predict_proba(x_test)[:, 1]
    xgb.save_model(str(out_dir / "models" / "xgboost.json"))

                  
    cat = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        random_seed=args.seed,
        depth=8,
        learning_rate=0.05,
        iterations=1500,
        l2_leaf_reg=3.0,
        verbose=200,
    )
    cat.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    p_val_cat = cat.predict_proba(x_val)[:, 1]
    p_test_cat = cat.predict_proba(x_test)[:, 1]
    cat.save_model(str(out_dir / "models" / "catboost.cbm"))

                    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_res = train_complex_nn(x_train, y_train, x_val, y_val, x_test, args.seed, device=device)
    p_val_nn = nn_res.val_proba
    p_test_nn = nn_res.test_proba
    torch.save(
        {
            "state_dict": nn_res.state_dict,
            "mean": nn_res.mean,
            "std": nn_res.std,
            "best_epoch": nn_res.best_epoch,
            "best_val_aucpr": nn_res.best_val_aucpr,
        },
        out_dir / "models" / "complex_mlp_meta.pt",
    )

    all_scores = {
        "xgboost": (p_val_xgb, p_test_xgb),
        "catboost": (p_val_cat, p_test_cat),
        "complex_mlp": (p_val_nn, p_test_nn),
    }

    summary_rows: List[Dict[str, object]] = []
    protocol_rows: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    full: Dict[str, object] = {}

    for name, (p_val, p_test) in all_scores.items():
        thr = select_threshold_recall_at_fpr(y_val, p_val, max_fpr=0.01)
        test_metrics = evaluate(y_test, p_test, thr["threshold"])
        test_metrics.update(
            {
                "model": name,
                "threshold": thr["threshold"],
                "val_recall_at_fpr_le_1pct": thr["val_recall"],
                "val_fpr_at_selected_threshold": thr["val_fpr"],
            }
        )
        summary_rows.append(test_metrics)
        p_rows = eval_by_slice(y_test, p_test, thr["threshold"], g_protocol, "protocol_hint", name)
        f_rows = eval_by_slice(y_test, p_test, thr["threshold"], g_family, "attack_family", name)
        protocol_rows.extend(p_rows)
        family_rows.extend(f_rows)
        full[name] = {"global_test": test_metrics, "protocol_slices": p_rows, "attack_family_slices": f_rows}

    summary_rows.sort(key=lambda r: float(r["f1"]), reverse=True)
    best_model = summary_rows[0]["model"]

    write_csv(
        out_dir / "metrics_summary.csv",
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
        out_dir / "slice_metrics_protocol.csv",
        protocol_rows,
        ["model", "protocol_hint", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )
    write_csv(
        out_dir / "slice_metrics_attack_family.csv",
        family_rows,
        ["model", "attack_family", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "seed": args.seed,
                "feature_columns": feature_cols,
                "dataset_sizes": {
                    "train": int(len(y_train)),
                    "val": int(len(y_val)),
                    "test": int(len(y_test)),
                    "train_attack_ratio": float(y_train.mean()),
                    "val_attack_ratio": float(y_val.mean()),
                    "test_attack_ratio": float(y_test.mean()),
                },
                "best_model_by_f1": best_model,
                "nn_best_epoch": nn_res.best_epoch,
                "nn_best_val_aucpr": nn_res.best_val_aucpr,
                "results": full,
            },
            f,
            indent=2,
        )

    with (out_dir / "RUN_SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Full-data GPU model training submitted/completed.\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Best model by F1: {best_model}\n")
        f.write(f"Rows train={len(y_train)} val={len(y_val)} test={len(y_test)}\n")
        f.write(f"Features={len(feature_cols)}\n")

    print(str(out_dir))


if __name__ == "__main__":
    main()
