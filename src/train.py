"""
train.py
========
Unified training loop for LSTM, TCN, and Transformer models.

Features
--------
- Early stopping (patience-based on validation loss)
- Exact training and inference timing
- Automatic checkpoint saving (best val loss)
- Experiment result logging to JSON
- Fixed random seed for reproducibility
"""

import os
import time
import json
import random
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from features import make_dataloaders, prepare_area_data


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int       = 100,
    lr: float           = 1e-3,
    patience: int       = 10,
    checkpoint_path: str = None,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """
    Train a model and return training history + timing statistics.

    Parameters
    ----------
    model           : nn.Module   The model to train
    train_loader    : DataLoader
    val_loader      : DataLoader
    n_epochs        : int         Maximum training epochs
    lr              : float       Learning rate for Adam
    patience        : int         Early stopping patience (epochs)
    checkpoint_path : str         Path to save best model weights
    device          : torch.device
    verbose         : bool

    Returns
    -------
    dict with keys:
        train_losses, val_losses, best_val_loss,
        best_epoch, total_train_time_s, avg_epoch_time_s
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
        # Note: verbose= kwarg removed in PyTorch 2.4+; LR changes logged manually below
    )
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_epoch     = 0
    no_improve     = 0
    epoch_times    = []

    t_train_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(-1).to(device)  # (batch, seq, 1)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred  = model(X_batch).squeeze(-1)
            loss  = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.unsqueeze(-1).to(device)
                y_batch = y_batch.to(device)
                pred    = model(X_batch).squeeze(-1)
                val_loss += criterion(pred, y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - t_epoch
        epoch_times.append(epoch_time)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_tag = f"  *** LR → {new_lr:.2e}" if new_lr < prev_lr else ""
            print(
                f"  Epoch {epoch:03d}/{n_epochs}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"lr={new_lr:.2e}  "
                f"time={epoch_time:.2f}s{lr_tag}"
            )

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best epoch={best_epoch})")
                break

    total_train_time = time.perf_counter() - t_train_start

    if verbose:
        print(f"\nTraining complete.")
        print(f"  Best val loss : {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"  Total train   : {total_train_time:.2f}s")
        print(f"  Avg per epoch : {np.mean(epoch_times):.2f}s")

    return {
        "train_losses":       train_losses,
        "val_losses":         val_losses,
        "best_val_loss":      best_val_loss,
        "best_epoch":         best_epoch,
        "total_train_time_s": total_train_time,
        "avg_epoch_time_s":   float(np.mean(epoch_times)),
        "epochs_run":         len(train_losses),
    }


# ---------------------------------------------------------------------------
# Inference / prediction
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    test_loader: DataLoader,
    scaler,
    device: torch.device,
    checkpoint_path: str = None,
) -> dict:
    """
    Run inference on test_loader and return predictions + timing.

    Returns
    -------
    dict with keys: preds_raw, targets_raw, inference_time_s, inference_time_per_sample_ms
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    model.eval()
    preds, targets = [], []

    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.unsqueeze(-1).to(device)
            out     = model(X_batch).squeeze(-1).cpu().numpy()
            preds.append(out)
            targets.append(y_batch.numpy())

    inference_time = time.perf_counter() - t_inf_start

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Inverse-transform to original scale
    preds_raw   = scaler.inverse_transform(preds)
    targets_raw = scaler.inverse_transform(targets)

    n_samples = len(preds)
    return {
        "preds_raw":                    preds_raw,
        "targets_raw":                  targets_raw,
        "inference_time_s":             inference_time,
        "inference_time_per_sample_ms": inference_time / n_samples * 1000,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_class,
    model_kwargs: dict,
    data: dict,
    n_epochs:   int   = 100,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    patience:   int   = 10,
    output_dir: str   = "experiments",
    model_name: str   = "model",
    verbose:    bool  = True,
) -> dict:
    """
    End-to-end training + evaluation for one model / hyperparameter config.

    Parameters
    ----------
    model_class  : class        One of LSTMForecaster, TCNForecaster, TransformerForecaster
    model_kwargs : dict         Constructor kwargs for model_class
    data         : dict         Output of features.prepare_area_data()
    ...

    Returns
    -------
    dict with training history, test predictions, and timings
    """
    set_seed(SEED)
    device = get_device()

    # Build data loaders
    train_loader, val_loader, test_loader = make_dataloaders(data, batch_size=batch_size)

    # Instantiate model
    model = model_class(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model : {model_name}  |  Params: {n_params:,}")
        print(f"{'='*60}")

    # Paths
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = os.path.join(output_dir, f"{model_name}_best.pt")

    # Train
    history = train_model(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=lr, patience=patience,
        checkpoint_path=ckpt, device=device, verbose=verbose,
    )

    # Predict
    pred_results = predict(model, test_loader, data["scaler"], device, checkpoint_path=ckpt)

    # Hardware info
    hw_info = {
        "platform":  platform.platform(),
        "processor": platform.processor(),
        "python":    platform.python_version(),
        "torch":     torch.__version__,
        "cuda":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    result = {
        "model_name":   model_name,
        "model_kwargs": model_kwargs,
        "n_params":     n_params,
        "square_id":    data["square_id"],
        "lookback":     data["lookback"],
        "lr":           lr,
        "batch_size":   batch_size,
        "patience":     patience,
        "history":      history,
        "preds_raw":    pred_results["preds_raw"].tolist(),
        "targets_raw":  pred_results["targets_raw"].tolist(),
        "test_index":   [str(t) for t in data["test_index"]],
        "inference_time_s":             pred_results["inference_time_s"],
        "inference_time_per_sample_ms": pred_results["inference_time_per_sample_ms"],
        "hardware":     hw_info,
    }

    # Save result JSON
    result_path = os.path.join(output_dir, f"{model_name}_sq{data['square_id']}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"Result saved → {result_path}")

    return result
