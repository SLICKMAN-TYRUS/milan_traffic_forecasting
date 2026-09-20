"""
evaluate.py
===========
Evaluation metrics and result visualisation for Section 4 of the assignment.

Provides
--------
  - MAE, MAPE, RMSE computation
  - Actual vs predicted plots (one per model × area)
  - Comparative summary tables
  - Failure case analysis plot
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
PALETTE = sns.color_palette("tab10")
MODEL_COLORS = {"LSTM": PALETTE[0], "TCN": PALETTE[1], "Transformer": PALETTE[2]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (skips near-zero actuals)."""
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_results_table(
    results: Dict[str, dict],
    square_id: int,
) -> pd.DataFrame:
    """
    Build a results DataFrame for one geographic area.

    Parameters
    ----------
    results   : dict mapping model_name → result dict (from train.run_experiment)
    square_id : int

    Returns
    -------
    pd.DataFrame with index = model names, columns = MAE, MAPE, RMSE
    """
    rows = []
    for model_name, res in results.items():
        y_true = np.array(res["targets_raw"])
        y_pred = np.array(res["preds_raw"])
        metrics = compute_all_metrics(y_true, y_pred)
        metrics["Train Time (s)"]     = res["history"]["total_train_time_s"]
        metrics["Inference (ms/samp)"] = res["inference_time_per_sample_ms"]
        rows.append(pd.Series(metrics, name=model_name))

    df = pd.DataFrame(rows).round(4)
    print(f"\nResults Table — Square {square_id}")
    print("-" * 60)
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Actual vs Predicted plots
# ---------------------------------------------------------------------------

def plot_predictions(
    results: Dict[str, dict],
    square_id: int,
    save_dir: str = None,
) -> None:
    """
    Plot actual vs predicted traffic — ONE FIGURE PER MODEL per area.

    The assignment (Section 4 item II) requires 9 plots total:
      3 models × 3 geographic areas = 9 separate figures.
    Each figure shows the full Dec 16–22 evaluation window with the
    actual series and the model's one-step-ahead predictions overlaid.
    """
    first_result = next(iter(results.values()))
    test_index   = pd.to_datetime(first_result["test_index"])
    y_true       = np.array(first_result["targets_raw"])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for model_name, res in results.items():
        y_pred  = np.array(res["preds_raw"])
        color   = MODEL_COLORS.get(model_name, PALETTE[0])
        metrics = compute_all_metrics(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(test_index, y_true,  color="black", linewidth=1.0,
                label="Actual",    alpha=0.90, zorder=3)
        ax.plot(test_index, y_pred,  color=color,   linewidth=1.0,
                label=f"{model_name} (predicted)", alpha=0.85,
                linestyle="--", zorder=3)

        # Shade weekends
        for d in pd.date_range(test_index[0], test_index[-1], freq="D"):
            if d.weekday() >= 5:
                ax.axvspan(d, d + pd.Timedelta(days=1),
                           alpha=0.07, color="grey", zorder=0)

        ax.set_ylabel("Internet Traffic (normalised activity)")
        ax.set_xlabel("Date (Dec 2013)")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(
            f"{model_name}  —  Square {square_id}  (Dec 16–22, 2013)\n"
            f"MAE={metrics['MAE']:.4f}   RMSE={metrics['RMSE']:.4f}   "
            f"MAPE={metrics['MAPE']:.2f}%",
            fontsize=11,
        )
        plt.tight_layout()

        if save_dir:
            fname = f"fig_pred_{model_name.lower()}_square{square_id}.png"
            fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight", dpi=150)
            print(f"Saved: {fname}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    results: Dict[str, dict],
    save_dir: str = None,
) -> None:
    """Plot train and validation loss curves for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        hist = res["history"]
        color = MODEL_COLORS.get(model_name, PALETTE[0])
        ax.plot(hist["train_losses"], label="Train", color=color,    linewidth=1.2)
        ax.plot(hist["val_losses"],   label="Val",   color=color,    linewidth=1.2, linestyle="--")
        ax.axvline(hist["best_epoch"] - 1, color="red", linewidth=0.8, linestyle=":", label="Best epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{model_name}\nBest epoch={hist['best_epoch']}, val={hist['best_val_loss']:.5f}")
        ax.legend(fontsize=8)

    fig.suptitle("Training and Validation Loss Curves", fontsize=12)
    plt.tight_layout()

    if save_dir:
        fname = "fig_training_curves.png"
        fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        print(f"Saved: {fname}")

    plt.show()


# ---------------------------------------------------------------------------
# Failure case analysis
# ---------------------------------------------------------------------------

def plot_failure_case(
    results: Dict[str, dict],
    square_id: int,
    save_dir: str = None,
) -> None:
    """
    Identify and plot the time window with the worst prediction error
    across all models — satisfying Section 4's failure case requirement.
    """
    first = next(iter(results.values()))
    test_index = pd.to_datetime(first["test_index"])
    y_true     = np.array(first["targets_raw"])

    # Compute absolute error per model
    errors = {}
    for name, res in results.items():
        errors[name] = np.abs(y_true - np.array(res["preds_raw"]))

    # Find the 6-hour window with the highest average error (across models)
    combined_error = np.mean(list(errors.values()), axis=0)
    window = 36    # 6 hours at 10-min intervals
    rolling_avg = np.convolve(combined_error, np.ones(window) / window, mode="valid")
    worst_start = int(np.argmax(rolling_avg))
    worst_end   = worst_start + window

    idx_slice   = test_index[worst_start:worst_end]
    y_slice     = y_true[worst_start:worst_end]

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        y_pred_slice = np.array(res["preds_raw"])[worst_start:worst_end]
        color = MODEL_COLORS.get(name, PALETTE[0])
        ax.plot(idx_slice, y_slice,       color="black", linewidth=1.2, label="Actual")
        ax.plot(idx_slice, y_pred_slice,  color=color,   linewidth=1.2, linestyle="--", label=name)
        err = mae(y_slice, y_pred_slice)
        ax.set_title(f"{name} — Worst Prediction Window (MAE={err:.3f})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("Traffic")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(f"Failure Case Analysis — Square {square_id}", fontsize=12)
    plt.tight_layout()

    if save_dir:
        fname = f"fig_failure_case_square{square_id}.png"
        fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        print(f"Saved: {fname}")

    plt.show()

    # Print context
    print(f"\nWorst prediction window: {idx_slice[0]} → {idx_slice[-1]}")
    print("Average absolute errors in this window:")
    for name in results:
        e = errors[name][worst_start:worst_end].mean()
        print(f"  {name}: {e:.4f}")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    all_results: Dict[int, Dict[str, dict]],
    save_dir: str = "report/figures",
) -> pd.DataFrame:
    """
    Run evaluation for all areas and all models.

    Parameters
    ----------
    all_results : dict  {square_id: {model_name: result_dict}}

    Returns
    -------
    pd.DataFrame  combined metrics table
    """
    all_tables = []

    for square_id, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"Evaluating Square {square_id}")
        print(f"{'='*60}")

        # Table
        table = build_results_table(results, square_id)
        table.insert(0, "Square", square_id)
        all_tables.append(table)

        # Plots
        plot_predictions(results, square_id, save_dir=save_dir)
        plot_failure_case(results, square_id, save_dir=save_dir)

    # Training curves (once, for the first area)
    first_area_results = next(iter(all_results.values()))
    plot_training_curves(first_area_results, save_dir=save_dir)

    combined = pd.concat(all_tables)
    combined.to_csv(os.path.join(save_dir, "..", "combined_metrics.csv"))
    print(f"\nCombined metrics saved → {os.path.join(save_dir, '..', 'combined_metrics.csv')}")

    return combined
