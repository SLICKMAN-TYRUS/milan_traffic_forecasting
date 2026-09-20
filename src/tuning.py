"""
tuning.py
=========
Hyperparameter tuning utilities.

Supports two modes:
  1. Manual iterative tuning  — documented in experiments/experiment_log.md
  2. Grid search              — automated sweep over a parameter grid

The assignment requires an *iterative, documented* approach where each experiment
informs the next.  Grid search is appropriate for non-critical sub-parameters
(e.g., batch size, dropout) once the key architecture decisions are made manually.
"""

import json
import time
import copy
import itertools
from pathlib import Path
from typing import Callable, Dict, List, Any

import numpy as np
import pandas as pd                       # FIX: imported at top, not bottom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from features import prepare_area_data, make_dataloaders
from train import train_model, set_seed, get_device
from evaluate import compute_all_metrics


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    model_class,
    param_grid: Dict[str, List[Any]],
    matrix,
    square_id: int,
    n_epochs: int   = 50,
    patience: int   = 7,
    save_dir: str   = "experiments/tuning",
    model_name: str = "model",
    max_configs: int = 20,
) -> pd.DataFrame:
    """
    Exhaustive (or capped) grid search over param_grid.

    Parameters
    ----------
    model_class  : class
    param_grid   : dict  {param_name: [val1, val2, …]}
                   Special keys handled separately: 'lookback', 'lr', 'batch_size'
                   All other keys are treated as model constructor kwargs.
    matrix       : pd.DataFrame  (T × N) processed traffic matrix
    square_id    : int
    n_epochs     : int   Max epochs per config
    patience     : int   Early stopping patience
    save_dir     : str   Where to save results
    model_name   : str   Prefix for output files
    max_configs  : int   Maximum number of configurations to evaluate

    Returns
    -------
    pd.DataFrame sorted by val MAE (ascending)
    """
    # FIX: deep-copy to avoid mutating the caller's dict
    grid = copy.deepcopy(param_grid)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()

    # Separate training-level params from model constructor params
    lookbacks   = grid.pop("lookback",    [144])
    lrs         = grid.pop("lr",          [1e-3])
    batch_sizes = grid.pop("batch_size",  [64])

    model_keys   = list(grid.keys())
    model_values = list(grid.values())
    model_combos = list(itertools.product(*model_values)) if model_keys else [()]

    all_combos = list(itertools.product(lookbacks, lrs, batch_sizes, model_combos))

    if len(all_combos) > max_configs:
        rng        = np.random.RandomState(42)
        idx        = rng.choice(len(all_combos), max_configs, replace=False)
        all_combos = [all_combos[i] for i in sorted(idx)]
        print(f"Grid search: sampling {max_configs} of {len(all_combos)} configs")

    print(f"Running {len(all_combos)} configurations for {model_name} on Square {square_id}")

    records = []
    for i, (lb, lr, bs, model_vals) in enumerate(all_combos, 1):
        model_kwargs = dict(zip(model_keys, model_vals)) if model_keys else {}
        config = {"lookback": lb, "lr": lr, "batch_size": bs, **model_kwargs}

        print(f"\n[{i}/{len(all_combos)}] {config}")
        set_seed(42)

        try:
            data = prepare_area_data(matrix, square_id, lookback=lb)
            train_loader, val_loader, _ = make_dataloaders(data, batch_size=bs)

            model   = model_class(input_size=1, **model_kwargs).to(device)
            history = train_model(
                model, train_loader, val_loader,
                n_epochs=n_epochs, lr=lr, patience=patience,
                device=device, verbose=False,
            )

            # Evaluate on validation set in original scale
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    p = model(X_b.unsqueeze(-1).to(device)).squeeze(-1).cpu().numpy()
                    val_preds.append(p)
                    val_true.append(y_b.numpy())

            val_preds = data["scaler"].inverse_transform(np.concatenate(val_preds))
            val_true  = data["scaler"].inverse_transform(np.concatenate(val_true))
            metrics   = compute_all_metrics(val_true, val_preds)

            record = {
                **config,
                **metrics,
                "best_val_loss": history["best_val_loss"],
                "best_epoch":    history["best_epoch"],
                "train_time_s":  history["total_train_time_s"],
                "epochs_run":    history["epochs_run"],
            }
            records.append(record)
            print(f"  → MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                  f"MAPE={metrics['MAPE']:.1f}%  epochs={history['epochs_run']}")

        except Exception as e:
            print(f"  ✗ Config failed: {e}")
            import traceback; traceback.print_exc()

    if not records:
        print("Warning: all configurations failed.")
        return pd.DataFrame()

    df  = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)
    out = Path(save_dir) / f"{model_name}_grid_search.csv"
    df.to_csv(out, index=False)
    print(f"\nGrid search complete. Results saved → {out}")
    print(f"Best config (val MAE={df.iloc[0]['MAE']:.4f}):")
    print(df.iloc[0].to_dict())

    return df


# ---------------------------------------------------------------------------
# Experiment logger (for manual iterative tuning)
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Lightweight logger for the manual iterative tuning process.

    Appends each experiment as a JSON line to a .jsonl file so the full
    history is readable as a DataFrame.  Also prints a concise summary so
    the rationale → result → next-step chain is visible at runtime.
    """

    def __init__(self, log_path: str = "experiments/experiment_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        experiment_id: str,
        model_name:    str,
        config:        dict,
        metrics:       dict,
        rationale:     str = "",
        next_steps:    str = "",
    ) -> None:
        """
        Log one experiment.

        Parameters
        ----------
        experiment_id : str   e.g. "LSTM_exp01"
        model_name    : str
        config        : dict  hyperparameters used
        metrics       : dict  MAE, MAPE, RMSE (validation or test)
        rationale     : str   Why this config was chosen
        next_steps    : str   What to try next based on the results
        """
        entry = {
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": experiment_id,
            "model_name":    model_name,
            "config":        config,
            "metrics":       metrics,
            "rationale":     rationale,
            "next_steps":    next_steps,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        mae_val  = metrics.get("MAE",  "-")
        rmse_val = metrics.get("RMSE", "-")
        mape_val = metrics.get("MAPE", "-")

        print(f"\n{'─'*55}")
        print(f"[Logged] {experiment_id}  ({model_name})")
        print(f"  Config    : {config}")
        if isinstance(mae_val, float):
            print(f"  Metrics   : MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  MAPE={mape_val:.1f}%")
        else:
            print(f"  Metrics   : {metrics}")
        if rationale:
            print(f"  Rationale : {rationale}")
        if next_steps:
            print(f"  Next      : {next_steps}")
        print(f"{'─'*55}")

    def to_dataframe(self) -> pd.DataFrame:
        """Load all logged experiments as a flat DataFrame."""
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.json_normalize(records)

    def print_summary(self) -> None:
        """Print a compact comparison of all logged experiments."""
        df = self.to_dataframe()
        if df.empty:
            print("No experiments logged yet.")
            return
        cols = ["experiment_id", "model_name",
                "metrics.MAE", "metrics.RMSE", "metrics.MAPE"]
        cols = [c for c in cols if c in df.columns]
        print("\nExperiment Summary")
        print("=" * 60)
        print(df[cols].to_string(index=False))
