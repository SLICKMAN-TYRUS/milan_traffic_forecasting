"""
tcn_model.py
============
Temporal Convolutional Network (TCN) for one-step-ahead traffic forecasting.

Architecture
------------
Input  : (batch, 1, lookback)   ← channels-first for Conv1d
         → N × ResidualBlock (dilated causal conv, weight norm, dropout, residual)
         → last timestep → Linear → scalar output

Each ResidualBlock:
    Conv1d (dilation d, kernel k, causal padding) → ReLU → Dropout
  → Conv1d (dilation d, kernel k, causal padding) → ReLU → Dropout
  → skip connection (1×1 conv if channel dimensions differ)

Dilation doubles each block: [1, 2, 4, 8, 16, …]
Receptive field = 1 + (kernel_size - 1) × 2 × (2^num_blocks - 1)

Rationale (from related work)
------------------------------
TCNs (Bai et al. 2018, "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling") demonstrate that dilated causal
convolutions match or outperform LSTMs on sequence modelling tasks while being
fully parallelisable during training.  For network traffic, the exponentially
growing receptive field allows the model to capture daily (144-step) and weekly
(1008-step) periodicities without the vanishing gradient issues of deep RNNs.
The causal structure ensures no future information leaks into predictions.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Causal conv block
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D convolution with causal (left-only) padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation,
            )
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove right-padding to enforce causality
        return out[:, :, :-self.padding] if self.padding > 0 else out


class ResidualBlock(nn.Module):
    """
    TCN residual block: two dilated causal conv layers + skip connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels,  out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1×1 conv for dimension matching in skip connection
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        return self.relu(self.net(x) + residual)


# ---------------------------------------------------------------------------
# Full TCN
# ---------------------------------------------------------------------------

class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for scalar one-step-ahead forecasting.

    Parameters
    ----------
    input_size   : int   Number of input channels (1 for univariate)
    num_channels : list  Number of channels per residual block
    kernel_size  : int   Convolution kernel width
    dropout      : float Dropout rate inside residual blocks
    """

    def __init__(
        self,
        input_size:   int   = 1,
        num_channels: list  = [32, 32, 64, 64],
        kernel_size:  int   = 3,
        dropout:      float = 0.2,
    ):
        super().__init__()

        layers = []
        in_ch  = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i          # exponential dilation: 1, 2, 4, 8 …
            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc      = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, lookback, 1)   ← (batch, seq, feat)

        Returns
        -------
        out : torch.Tensor  shape (batch, 1)
        """
        # TCN expects channels-first: (batch, feat, seq)
        x = x.permute(0, 2, 1)              # (batch, 1, lookback)
        x = self.network(x)                 # (batch, out_ch, lookback)
        x = x[:, :, -1]                     # last timestep (causal)
        return self.fc(x)                   # (batch, 1)


# ---------------------------------------------------------------------------
# Default hyperparameter grid for tuning
# ---------------------------------------------------------------------------
TCN_PARAM_GRID = {
    "num_channels": [
        [32, 32, 64, 64],
        [64, 64, 64, 64],
        [32, 64, 128, 128],
    ],
    "kernel_size":  [3, 5, 7],
    "dropout":      [0.1, 0.2, 0.3],
    "lookback":     [72, 144, 288],
    "lr":           [1e-3, 5e-4, 1e-4],
    "batch_size":   [32, 64],
}
