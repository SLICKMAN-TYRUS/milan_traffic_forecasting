"""
lstm_model.py
=============
Stacked LSTM for one-step-ahead Internet traffic forecasting.

Architecture
------------
Input  : (batch, lookback, 1)
         → LSTM layer 1  (hidden_size, dropout between layers)
         → LSTM layer 2  (hidden_size)
         → Fully connected head  → scalar output

Rationale (from related work)
------------------------------
LSTMs are the canonical recurrent baseline for network traffic forecasting.
Their gating mechanism (input, forget, output gates) allows them to selectively
retain information over hundreds of timesteps, making them well-suited to the
daily and weekly periodicities visible in the Milan data.  Prior work (e.g.,
Huang et al. 2017; Vinayakumar et al. 2017) consistently finds that LSTM
outperforms simple RNN and ARIMA baselines for cellular traffic prediction.
Two stacked layers allow the model to learn both low-level temporal patterns
(hour-scale) and higher-level seasonal structure (day/week-scale).
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM followed by a linear output head.

    Parameters
    ----------
    input_size  : int   Number of features per timestep (1 for univariate)
    hidden_size : int   Number of hidden units per LSTM cell
    num_layers  : int   Number of stacked LSTM layers
    dropout     : float Dropout probability between LSTM layers (0 = none)
    """

    def __init__(
        self,
        input_size:  int   = 1,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
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
        # h0, c0 initialised to zeros by default
        lstm_out, _ = self.lstm(x)         # (batch, seq_len, hidden_size)
        last_hidden  = lstm_out[:, -1, :]  # take final timestep
        out          = self.fc(last_hidden) # (batch, 1)
        return out


# ---------------------------------------------------------------------------
# Default hyperparameter grid for tuning
# ---------------------------------------------------------------------------
LSTM_PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers":  [1, 2],
    "dropout":     [0.0, 0.2, 0.3],
    "lookback":    [72, 144, 288],       # 12h, 24h, 48h
    "lr":          [1e-3, 5e-4, 1e-4],
    "batch_size":  [32, 64],
}
