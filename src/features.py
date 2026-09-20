"""
features.py
===========
Sequence construction, train/val/test splitting, and normalisation
for one-step-ahead Internet traffic forecasting.

The forecasting task:
  Given a history vector  x[t-L : t]  of length L (the look-back window),
  predict  x[t+1]  for a single geographic area.

Splits (by time, never shuffled):
  Train  : all data before the evaluation week  (Dec 16–22)
  Test   : December 16–22 (one-step-ahead rolling forecast)
  Val    : last 10% of the training portion (for hyperparameter tuning)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Evaluation week required by the assignment
EVAL_START = "2013-12-16"
EVAL_END   = "2013-12-22 23:59:59"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class AreaScaler:
    """
    Wraps sklearn MinMaxScaler for a single time series.
    Fit only on training data; transform train + test.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self._fitted = False

    def fit(self, train_values: np.ndarray) -> "AreaScaler":
        self.scaler.fit(train_values.reshape(-1, 1))
        self._fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()."
        return self.scaler.transform(values.reshape(-1, 1)).flatten()

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    series: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D time series into (X, y) supervised learning pairs.

    X[i] = series[i : i+lookback]     shape (lookback,)
    y[i] = series[i+lookback]          scalar

    Parameters
    ----------
    series   : 1-D np.ndarray of normalised traffic values
    lookback : number of past time steps to use as input

    Returns
    -------
    X : np.ndarray  shape (N, lookback)
    y : np.ndarray  shape (N,)
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train / Val / Test split for one geographic area
# ---------------------------------------------------------------------------

def prepare_area_data(
    matrix: pd.DataFrame,
    square_id: int,
    lookback: int = 144,               # 1 day of 10-min intervals
    val_fraction: float = 0.1,
) -> dict:
    """
    Full pipeline for one geographic area:
      1. Extract series
      2. Split into train/val/test by date
      3. Fit scaler on train, transform all splits
      4. Build supervised (X, y) sequences

    Parameters
    ----------
    matrix       : (T × 10000) processed traffic matrix with DatetimeIndex
    square_id    : target geographic cell
    lookback     : sequence length L
    val_fraction : fraction of training data reserved for validation

    Returns
    -------
    dict with keys:
      series_raw, train_raw, val_raw, test_raw   — raw pd.Series
      scaler                                      — fitted AreaScaler
      X_train, y_train, X_val, y_val             — normalised arrays
      X_test,  y_test                            — normalised arrays
      test_index                                 — DatetimeIndex for test targets
    """
    series = matrix[square_id]

    # --- Split by date ---
    test_mask  = (series.index >= EVAL_START) & (series.index <= EVAL_END)
    train_full = series[~test_mask]
    test_series = series[test_mask]

    val_size  = int(len(train_full) * val_fraction)
    val_series   = train_full.iloc[-val_size:]
    train_series = train_full.iloc[:-val_size]

    # --- Fit scaler on train only ---
    scaler = AreaScaler()
    scaler.fit(train_series.values)

    train_norm = scaler.transform(train_series.values)
    val_norm   = scaler.transform(val_series.values)
    test_norm  = scaler.transform(test_series.values)

    # --- Build sequences ---
    X_train, y_train = build_sequences(train_norm, lookback)
    X_val,   y_val   = build_sequences(val_norm,   lookback)
    # For test: prepend the last `lookback` steps from val so we can make
    # predictions starting from the very first test timestep
    test_with_context = np.concatenate([val_norm[-lookback:], test_norm])
    X_test, y_test    = build_sequences(test_with_context, lookback)

    # Align test targets with DatetimeIndex
    test_index = test_series.index

    return {
        # Raw series
        "series_raw":   series,
        "train_raw":    train_series,
        "val_raw":      val_series,
        "test_raw":     test_series,
        # Scaler
        "scaler":       scaler,
        # Normalised arrays
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        # Metadata
        "lookback":    lookback,
        "test_index":  test_index,
        "square_id":   square_id,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    class TrafficDataset(Dataset):
        """PyTorch Dataset wrapping (X, y) numpy arrays."""

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def make_dataloaders(
        data: dict,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Tuple["DataLoader", "DataLoader", "DataLoader"]:
        """
        Create PyTorch DataLoaders from the output of prepare_area_data().

        Returns
        -------
        train_loader, val_loader, test_loader
        """
        train_ds = TrafficDataset(data["X_train"], data["y_train"])
        val_ds   = TrafficDataset(data["X_val"],   data["y_val"])
        test_ds  = TrafficDataset(data["X_test"],  data["y_test"])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

except ImportError:
    print("PyTorch not found — DataLoader utilities unavailable.")
