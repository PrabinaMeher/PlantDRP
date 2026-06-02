#!/usr/bin/env python3
"""
PlantDRP — Training & Evaluation Pipeline
==========================================
Runs GridSearchCV (hyperparameter tuning) on the training set,
performs outer 5-fold cross-validation with the best params,
retrains on the full training set, evaluates on an independent
test set, and saves the final model + all results.

Usage
--
    python train_plantdrp.py \
        --train embeddings/train_embeddings.csv \
        --test  embeddings/test_embeddings.csv  \
        --out   results/dataset1               \
        --name  "Dataset-1"

The --out flag is treated as an output *directory*.
All CSVs, plots, and the saved model go inside it.
"""

import argparse
import os
import sys
import time
import warnings
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")


# 1. DATA LOADING


def load_embeddings(path: str):
    """
    Load a CSV produced by the embedding script.
    Expected format: first column = label (0/1), remaining columns = features.
    """
    df = pd.read_csv(path, header=None)

    if df.shape[1] < 2:
        sys.exit(f"[ERROR] {path} must have at least 2 columns (label + features).")

    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(np.float32)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"  Loaded {path}  →  {len(y)} samples  (pos={pos}, neg={neg})")
    return X, y



# 2. METRICS


METRIC_COLS = ["Sn", "Sp", "Pre", "Acc", "MCC", "F1", "AUROC", "AUPRC"]

def compute_metrics(y_true, y_pred, y_prob):
    """Return all 8 performance metrics as a list."""
    cm = confusion_matrix(y_true, y_pred)

    # Handle edge case: binary confusion matrix may not be 2×2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Only one class present in predictions — degenerate fold
        tn = fp = fn = tp = 0

    sn  = tp / (tp + fn) if (tp + fn) else 0.0      # sensitivity / recall
    sp  = tn / (tn + fp) if (tn + fp) else 0.0      # specificity
    pre = precision_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    f1  = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.0

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = 0.0

    return [sn, sp, pre, acc, mcc, f1, auroc, auprc]


def metrics_row(tag, y_true, y_pred, y_prob):
    return [tag] + compute_metrics(y_true, y_pred, y_prob)



# 3. MODEL BUILDING


def build_pipeline():
    """StandardScaler → SVC with probability estimates."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=True, random_state=42))
    ])


def hyperparameter_search(X, y, cv_folds=5, n_jobs=-1, verbose=1):
    """
    Inner GridSearchCV: finds the best (C, gamma, kernel) combo.
    Uses stratified k-fold on the training set.
    Scoring: ROC-AUC (robust to class imbalance).
    Returns the fitted best estimator and its parameter dict.
    """
    param_grid = {
        "svm__C":      [0.1, 1, 10, 100],
        "svm__gamma":  ["scale", 0.1, 0.01, 0.001],
        "svm__kernel": ["rbf", "linear"]
    }

    grid = GridSearchCV(
        build_pipeline(),
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True          # refit best params on the full X passed in
    )

    print("\n[STEP 1] Grid search (inner CV) …")
    t0 = time.time()
    grid.fit(X, y)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV AUC : {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_



# 4. OUTER CROSS-VALIDATION


def outer_cross_validation(best_params, X, y, folds=5):
    """
    Outer 5-fold CV using the best hyperparameters found in the inner search.
    Each fold trains from scratch — no data leakage.
    Returns per-fold rows and the mean ± std summary row.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []

    print(f"\n[STEP 2] Outer {folds}-fold cross-validation …")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr,  y_tr  = X[tr_idx],  y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_pipeline()
        model.set_params(**best_params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        row = metrics_row(f"CV_Fold_{fold}", y_val, y_pred, y_prob)
        rows.append(row)
        vals = row[1:]
        print(f"  Fold {fold}: Acc={vals[3]:.4f}  MCC={vals[4]:.4f}  "
              f"AUROC={vals[6]:.4f}  AUPRC={vals[7]:.4f}")

    # Mean and std across folds
    arr = np.array([r[1:] for r in rows], dtype=float)
    mean_row = ["CV_Mean"] + list(np.mean(arr, axis=0))
    std_row  = ["CV_Std"]  + list(np.std(arr,  axis=0))

    m = mean_row[1:]
    print(f"\n  CV mean   Acc={m[3]:.4f}  MCC={m[4]:.4f}  "
          f"AUROC={m[6]:.4f}  AUPRC={m[7]:.4f}")

    return rows, mean_row, std_row



# 5. FINAL MODEL — retrain on full train set


def train_final_model(best_params, X_train, y_train):
    """Retrain with best params on ALL training data."""
    print("\n[STEP 3] Retraining final model on full training set …")
    model = build_pipeline()
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    print("  Done.")
    return model



# 6. INDEPENDENT TEST EVALUATION


def evaluate_test(model, X_test, y_test):
    print("\n[STEP 4] Evaluating on independent test set …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    row    = metrics_row("Test", y_test, y_pred, y_prob)
    vals   = row[1:]
    print(f"  Test  →  Acc={vals[3]:.4f}  MCC={vals[4]:.4f}  "
          f"AUROC={vals[6]:.4f}  AUPRC={vals[7]:.4f}")
    return row, y_prob, y_pred



# 7. PLOTTING


COLORS = {
    "roc_train": "#2563EB",
    "pr_train":  "#16A34A",
    "roc_test":  "#DC2626",
    "pr_test":   "#D97706",
    "diagonal":  "#9CA3AF",
    "fold":      "#93C5FD",
}

def plot_roc_pr(model, X_train, y_train, X_test, y_test, out_dir, dataset_name):
    """
    Saves two publication-quality plots:
      1. ROC curve (train + test)
      2. Precision-Recall curve (train + test)
    """
    # Probabilities
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]

    fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_train)
    fpr_te, tpr_te, _ = roc_curve(y_test,  prob_test)
    auc_tr = roc_auc_score(y_train, prob_train)
    auc_te = roc_auc_score(y_test,  prob_test)

    pre_tr, rec_tr, _ = precision_recall_curve(y_train, prob_train)
    pre_te, rec_te, _ = precision_recall_curve(y_test,  prob_test)
    ap_tr = average_precision_score(y_train, prob_train)
    ap_te = average_precision_score(y_test,  prob_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"PlantDRP — {dataset_name}", fontsize=13, fontweight="bold", y=1.01)

    #  ROC 
    ax = axes[0]
    ax.plot(fpr_tr, tpr_tr, color=COLORS["roc_train"],
            lw=2, label=f"Train (AUC = {auc_tr:.3f})")
    ax.plot(fpr_te, tpr_te, color=COLORS["roc_test"],
            lw=2, linestyle="--", label=f"Test  (AUC = {auc_te:.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["diagonal"], lw=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    #  Precision-Recall 
    ax = axes[1]
    ax.plot(rec_tr, pre_tr, color=COLORS["pr_train"],
            lw=2, label=f"Train (AP = {ap_tr:.3f})")
    ax.plot(rec_te, pre_te, color=COLORS["pr_test"],
            lw=2, linestyle="--", label=f"Test  (AP = {ap_te:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cv_bars(cv_rows, out_dir, dataset_name):
    """Bar chart of per-fold performance for 4 key metrics."""
    folds  = [r[0] for r in cv_rows]
    arr    = np.array([r[1:] for r in cv_rows], dtype=float)
    keys   = ["Acc", "MCC", "AUROC", "AUPRC"]
    idxs   = [3, 4, 6, 7]
    colors = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]

    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (key, idx, col) in enumerate(zip(keys, idxs, colors)):
        vals = arr[:, idx]
        ax.bar(x + i * width, vals, width, label=key, color=col, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(folds, rotation=15, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Per-fold CV performance — {dataset_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "cv_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



# 8. SAVING OUTPUTS


def save_model(model, best_params, out_dir, dataset_name):
    payload = {
        "model":        model,
        "best_params":  best_params,
        "dataset_name": dataset_name,
    }
    path = os.path.join(out_dir, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"  Saved: {path}")
    return path


def save_results(cv_rows, mean_row, std_row, test_row, best_params,
                 best_cv_auc, out_dir, dataset_name):
    cols = ["Set"] + METRIC_COLS
    rows = cv_rows + [mean_row, std_row, test_row]
    df   = pd.DataFrame(rows, columns=cols)

    # Round floats for readability
    df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce").round(4)

    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Also save a JSON summary (easy to parse in the Streamlit app)
    summary = {
        "dataset":      dataset_name,
        "best_params":  best_params,
        "best_cv_auc":  round(float(best_cv_auc), 4),
        "cv_mean":      dict(zip(METRIC_COLS, [round(v, 4) for v in mean_row[1:]])),
        "cv_std":       dict(zip(METRIC_COLS, [round(v, 4) for v in std_row[1:]])),
        "test":         dict(zip(METRIC_COLS, [round(v, 4) for v in test_row[1:]])),
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    return df



# 9. MAIN


def parse_args():
    p = argparse.ArgumentParser(
        description="PlantDRP — SVM training pipeline with hyperparameter tuning")
    p.add_argument("--train",   required=True,  help="Path to train embeddings CSV")
    p.add_argument("--test",    required=True,  help="Path to test embeddings CSV")
    p.add_argument("--out",     default="results/run1",
                                help="Output directory (created if absent)")
    p.add_argument("--name",    default="PlantDRP",
                                help="Dataset name (used in plots and summary)")
    p.add_argument("--cv",      type=int, default=5,
                                help="Number of outer CV folds (default: 5)")
    p.add_argument("--inner-cv",type=int, default=5, dest="inner_cv",
                                help="Folds for inner GridSearch (default: 5)")
    p.add_argument("--jobs",    type=int, default=-1,
                                help="Parallel jobs for GridSearch (default: -1 = all CPUs)")
    p.add_argument("--verbose", type=int, default=1,
                                help="GridSearch verbosity (0 = silent, 2 = per-candidate)")
    return p.parse_args()


def main():
    args = parse_args()

    #  Setup output directory 
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  PlantDRP Training Pipeline")
    print(f"  Dataset : {args.name}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}\n")

    #  Load data 
    print("[DATA]")
    X_train, y_train = load_embeddings(args.train)
    X_test,  y_test  = load_embeddings(args.test)

    #  Hyperparameter search (inner CV) 
    best_model, best_params, best_cv_auc = hyperparameter_search(
        X_train, y_train,
        cv_folds=args.inner_cv,
        n_jobs=args.jobs,
        verbose=args.verbose
    )

    # Convert Pipeline param names back to plain SVC param names
    # (needed when we rebuild the pipeline per-fold in outer CV)
    plain_params = {
        k.replace("svm__", ""): v
        for k, v in best_params.items()
    }
    pipeline_params = best_params   # keep the svm__ prefix form for set_params

    #  Outer CV 
    cv_rows, mean_row, std_row = outer_cross_validation(
        pipeline_params, X_train, y_train, folds=args.cv)

    #  Final model 
    final_model = train_final_model(pipeline_params, X_train, y_train)

    #  Test evaluation 
    test_row, y_prob_test, y_pred_test = evaluate_test(
        final_model, X_test, y_test)

    #  Save all outputs 
    print("\n[OUTPUT]")

    df_results = save_results(
        cv_rows, mean_row, std_row, test_row,
        pipeline_params, best_cv_auc, out_dir, args.name)

    model_path = save_model(final_model, pipeline_params, out_dir, args.name)

    plot_roc_pr(final_model, X_train, y_train, X_test, y_test,
                out_dir, args.name)
    plot_cv_bars(cv_rows, out_dir, args.name)

    #  Print final summary 
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n  Model saved → {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
