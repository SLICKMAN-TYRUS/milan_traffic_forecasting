"""
transformer_model.py
====================
Time-series Transformer for one-step-ahead Internet traffic forecasting.

Architecture
------------
Input  : (batch, lookback, 1)
         → Linear projection → d_model dimensions
         → Positional Encoding (sinusoidal)
         → N × TransformerEncoderLayer (multi-head self-attention + FFN)
         → last position → Linear → scalar output

Rationale (from related work)
------------------------------
Transformer-based models (Vaswani et al. 2017) have shown strong performance
on sequential forecasting tasks (e.g., Informer — Zhou et al. 2021; PatchTST —
Nie et al. 2023).  Unlike LSTMs, self-attention directly models pairwise
dependencies between any two timesteps regardless of distance, making it
theoretically well-suited to capturing both short-term (hourly) and long-range
(weekly) periodicities simultaneously.  The encoder-only design is appropriate
for one-step-ahead forecasting where we do not need an autoregressive decoder.

Key difference from LSTM and TCN:
  Attention is *non-causal by default* over the input window — the model
  attends globally across the lookback window.  This is a fundamentally
  different inductive bias from the sequential hidden-state (LSTM) or
  local-receptive-field (TCN) approaches, making it the third meaningfully
  distinct architecture in our comparison.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Forecaster
# ---------------------------------------------------------------------------

class TransformerForecaster(nn.Module):
    """
    Encoder-only Transformer for one-step-ahead forecasting.

    Parameters
    ----------
    input_size  : int   Input feature dimension (1 for univariate)
    d_model     : int   Internal embedding dimension
    nhead       : int   Number of attention heads  (must divide d_model)
    num_layers  : int   Number of TransformerEncoder layers
    dim_feedfwd : int   Feedforward hidden size inside each encoder layer
    dropout     : float Dropout rate
    """

    def __init__(
        self,
        input_size:  int   = 1,
        d_model:     int   = 64,
        nhead:       int   = 4,
        num_layers:  int   = 2,
        dim_feedfwd: int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Project raw input features → d_model
        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers, enable_nested_tensor=False
   )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, input_size)

        Returns
        -------
        out : torch.Tensor  shape (batch, 1)
        """
        x = self.input_projection(x)          # (batch, seq, d_model)
        x = self.pos_encoding(x)              # adds positional info
        x = self.transformer_encoder(x)       # (batch, seq, d_model)
        x = x[:, -1, :]                       # last position representation
        return self.output_head(x)            # (batch, 1)


# ---------------------------------------------------------------------------
# Default hyperparameter grid for tuning
# ---------------------------------------------------------------------------
TRANSFORMER_PARAM_GRID = {
    "d_model":     [32, 64, 128],
    "nhead":       [2, 4, 8],
    "num_layers":  [1, 2, 3],
    "dim_feedfwd": [64, 128, 256],
    "dropout":     [0.1, 0.2],
    "lookback":    [72, 144, 288],
    "lr":          [1e-3, 5e-4, 1e-4],
    "batch_size":  [32, 64],
}
