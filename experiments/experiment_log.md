# Experiment Log — Milan Traffic Forecasting

This document records the iterative hyperparameter tuning process for each model.
Each experiment documents: the parameters used, the results obtained on the
**validation set** (in original traffic units), and the reasoning behind subsequent changes.

All tuning was performed on **Square 5161** (highest total traffic).
Best configurations were then applied to all three target areas for final evaluation.

Hardware: Windows 11, CPU-only, PyTorch 2.14.0, seed=42.

---

## Target Area

- **Primary tuning area**: Square 5161 (total traffic = 12,740,060)
- **Final evaluation areas**: Square 5161, Square 5059, Square 5259

---

## Model 1: LSTM

### Experiment LSTM-01

**Rationale:** Baseline configuration from literature. Hidden size 64, 2 layers,
lookback of 1 day (144 steps at 10-min intervals) — the standard starting point in
cellular traffic forecasting (Huang et al. 2017; Vinayakumar et al. 2017).

| Parameter    | Value |
|-------------|-------|
| hidden_size  | 64    |
| num_layers   | 2     |
| dropout      | 0.2   |
| lookback     | 144   |
| lr           | 1e-3  |
| batch_size   | 64    |
| epochs (max) | 100   |

**Results (validation set, original units):**

| Metric | Value    |
|--------|----------|
| MAE    | 83.56    |
| MAPE   | 8.945%   |
| RMSE   | 124.46   |

**Observations:** Best validation checkpoint at epoch 30; early stopping at epoch 40.
The model converges cleanly with no sign of overfitting. Val loss 0.000214. The
1-day lookback appears sufficient to exploit the dominant daily periodicity confirmed
in the ACF. This is a strong baseline.

**Next steps:** Test whether the significant autocorrelation visible at lag 288 (2 days,
per ACF analysis) can be exploited by expanding hidden_size to 128 and lookback to
288. Expect higher training time due to doubled sequence length.

---

### Experiment LSTM-02

**Rationale:** ACF showed strong autocorrelation at lag 288. Expanding lookback
from 144 → 288 steps and increasing hidden_size 64 → 128 to test whether
two-day context improves prediction. Reducing LR slightly to 5e-4 for stability with
the larger model.

| Parameter    | Value |
|-------------|-------|
| hidden_size  | 128   |
| num_layers   | 2     |
| dropout      | 0.2   |
| lookback     | 288   |
| lr           | 5e-4  |
| batch_size   | 64    |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 84.79    |
| MAPE   | 8.613%   |
| RMSE   | 128.50   |

**Observations:** Marginally *worse* than Exp 1 (MAE: 83.56 → 84.79) despite doubling
model capacity. Training time: **24,685 seconds** (~6.9 hours) — prohibitively slow
on CPU due to the doubled sequence length increasing LSTM recurrence cost quadratically.
The larger model (207,489 params) appears to overfit relative to the per-area training
set size.

**Next steps:** Try dropout=0.3 at the same configuration to attempt regularisation.
Exp 1 config remains the current favourite.

---

### Experiment LSTM-03

**Rationale:** Increasing dropout 0.2 → 0.3 to regularise the larger model seen
to overfit in Exp 2.

| Parameter    | Value |
|-------------|-------|
| hidden_size  | 128   |
| num_layers   | 2     |
| dropout      | 0.3   |
| lookback     | 288   |
| lr           | 5e-4  |
| batch_size   | 64    |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 93.81    |
| MAPE   | 11.616%  |
| RMSE   | 133.46   |

**Observations:** Higher dropout degrades performance further (MAE: 93.81 vs 84.79
in Exp 2). The model is not overfitting in the conventional sense — the issue is that
the doubled lookback and larger architecture do not add predictive value beyond
what the 1-day window already captures, and the additional regularisation only
removes signal.

**Decision: LSTM-01 selected as final config** (hidden_size=64, lookback=144, lr=1e-3).
Rationale: lowest validation MAE (83.56), fastest training (~980s), cleanest convergence.

---

## Model 2: TCN

### Experiment TCN-01

**Rationale:** Standard TCN from Bai et al. (2018): 4 residual blocks with channels
[32, 32, 64, 64], kernel size 3, exponential dilation. Receptive field =
1 + 2×(3−1)×(1+2+4+8) = **61 steps (~10 hours)**. This does not yet cover the full
daily cycle (144 steps); baseline intended to confirm the model trains stably.

| Parameter    | Value            |
|-------------|------------------|
| num_channels | [32, 32, 64, 64] |
| kernel_size  | 3                |
| dropout      | 0.2              |
| lookback     | 144              |
| lr           | 1e-3             |
| batch_size   | 64               |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 80.95    |
| MAPE   | 8.348%   |
| RMSE   | 119.03   |

**Observations:** Already outperforms LSTM-01 (MAE 80.95 vs 83.56). Training converges
quickly (best epoch 21) due to fully parallelisable convolutions. Receptive field
of 61 steps is shorter than the daily cycle — expanding it should help.

**Next steps:** Increase kernel size 3 → 5 and widen to uniform 64 channels.
New RF = 1 + 2×(5−1)×(1+2+4+8) = **121 steps (~20 hours)**, covering ~84% of
the daily cycle.

---

### Experiment TCN-02

**Rationale:** Wider kernel (k=5) extends receptive field to 121 steps (~20h).
Uniform [64,64,64,64] channels increases model capacity.

| Parameter    | Value            |
|-------------|------------------|
| num_channels | [64, 64, 64, 64] |
| kernel_size  | 5                |
| dropout      | 0.2              |
| lookback     | 144              |
| lr           | 1e-3             |
| batch_size   | 64               |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 80.87    |
| MAPE   | 8.882%   |
| RMSE   | 118.23   |

**Observations:** Marginal improvement over Exp 1 (MAE: 80.87 vs 80.95, RMSE: 118.23
vs 119.03). The wider kernel consistently improves RMSE, confirming the benefit of
extending the receptive field toward the daily cycle. Training speed remains fast
(best epoch 21, total ~1,223s). This is the best TCN configuration found so far.

**Next steps:** Test whether extending lookback to 288 or reducing LR further provides
additional gain. Expect diminishing returns given the marginal Exp 1→2 improvement.

---

### Experiment TCN-03

**Rationale:** Extended lookback to 288 and reduced LR to 5e-4 and increased
dropout to 0.3 to test whether additional context and regularisation help.

| Parameter    | Value            |
|-------------|------------------|
| num_channels | [64, 64, 64, 64] |
| kernel_size  | 5                |
| dropout      | 0.3              |
| lookback     | 288              |
| lr           | 5e-4             |
| batch_size   | 64               |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 82.12    |
| MAPE   | 8.359%   |
| RMSE   | 120.46   |

**Observations:** Slight degradation vs Exp 2 (MAE: 82.12 vs 80.87). The 2-day
lookback does not benefit the TCN — its fixed receptive field of 121 steps already
extracts the relevant daily signal; providing additional context the model cannot
attend to within its architectural inductive bias adds noise rather than signal.

**Decision: TCN-02 selected as final config** (channels=[64,64,64,64], k=5, lookback=144, lr=1e-3).
Best validation MAE (80.87), fastest convergence, no benefit from larger context.

---

## Model 3: Transformer

### Experiment TF-01

**Rationale:** Small, conservative starting configuration (d_model=64, 2 layers,
nhead=4) with a low learning rate (1e-4) to ensure stable training. Pre-LN
normalisation (Xiong et al., 2020) used throughout for stability.

| Parameter    | Value |
|-------------|-------|
| d_model      | 64    |
| nhead        | 4     |
| num_layers   | 2     |
| dim_feedfwd  | 128   |
| dropout      | 0.1   |
| lookback     | 144   |
| lr           | 1e-4  |
| batch_size   | 32    |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 151.87   |
| MAPE   | 26.847%  |
| RMSE   | 191.93   |

**Observations:** Very poor performance — MAE of 151.87 is roughly 80% worse than
LSTM-01 (83.56). The model is under-capacity: d_model=64 with 2 layers is
insufficient for this task. High MAPE (26.85%) suggests the model fails entirely
on low-traffic intervals. The low LR (1e-4) also likely slows convergence
beyond the early-stopping patience.

**Next steps:** Scale capacity substantially — d_model 64→128, 2→3 layers,
ffn 128→256. Increase LR to 5e-4 given the model did not appear to diverge.

---

### Experiment TF-02

**Rationale:** Substantially increase capacity (d_model=128, 3 layers, ffn=256)
and raise LR to 5e-4. This creates a more powerful model capable of learning
the complex daily periodic structure.

| Parameter    | Value |
|-------------|-------|
| d_model      | 128   |
| nhead        | 4     |
| num_layers   | 3     |
| dim_feedfwd  | 256   |
| dropout      | 0.1   |
| lookback     | 144   |
| lr           | 5e-4  |
| batch_size   | 32    |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 109.99   |
| MAPE   | 13.522%  |
| RMSE   | 154.80   |

**Observations:** Major improvement over Exp 1 (MAE: 109.99 vs 151.87), confirming
that capacity and learning rate are both essential. However, all metrics still
substantially lag LSTM and TCN. The model's 406,017 parameters relative to the
per-area training data volume likely creates conditions favouring overfitting.
Training cost: 12,364s — approximately 13× slower than LSTM.

**Next steps:** Test higher dropout (0.2) to assess whether regularisation can
close the performance gap. Exp 2 is current best Transformer config.

---

### Experiment TF-03

**Rationale:** Increasing dropout 0.1 → 0.2 to reduce suspected overfitting.

| Parameter    | Value |
|-------------|-------|
| d_model      | 128   |
| nhead        | 4     |
| num_layers   | 3     |
| dim_feedfwd  | 256   |
| dropout      | 0.2   |
| lookback     | 144   |
| lr           | 5e-4  |
| batch_size   | 32    |

**Results (validation set):**

| Metric | Value    |
|--------|----------|
| MAE    | 127.36   |
| MAPE   | 14.556%  |
| RMSE   | 178.64   |

**Observations:** Higher dropout significantly degrades performance (MAE: 127.36
vs 109.99). This suggests the Transformer at Exp 2 scale is not overfitting —
the underperformance relative to LSTM/TCN is structural: the O(L²) self-attention
may attend to irrelevant cross-timestep dependencies rather than the local
short-range and daily periodic structure that both recurrent and convolutional
architectures more naturally prioritise on short univariate sequences
(consistent with Zeng et al., 2023).

**Decision: TF-02 selected as final config** (d_model=128, 3 layers, lr=5e-4).
Best validation MAE (109.99) despite high parameter count; best achievable on
this dataset with this architecture.

---

## Summary: All Nine Tuning Experiments

| Exp ID     | Model       | Key Changes vs Previous            | Val MAE | Val RMSE | Decision |
|-----------|-------------|-------------------------------------|---------|----------|----------|
| LSTM-01   | LSTM        | Baseline: h=64, L=2, lb=144, lr=1e-3  | 83.56  | 124.46   | **Best LSTM — selected as final** |
| LSTM-02   | LSTM        | h=128, lb=288, lr=5e-4 (24,685s train) | 84.79  | 128.50   | Worse + prohibitively slow — reverted |
| LSTM-03   | LSTM        | dropout=0.3 at LSTM-02 config       | 93.81   | 133.46   | Worse — higher dropout hurts at this scale |
| TCN-01    | TCN         | ch=[32,32,64,64], k=3, lb=144       | 80.95   | 119.03   | Strong baseline; RF=61 steps (~10h) |
| TCN-02    | TCN         | ch=[64,64,64,64], k=5; RF=121 steps | 80.87   | 118.23   | **Best TCN — selected as final** |
| TCN-03    | TCN         | lb=288, dropout=0.3, lr=5e-4        | 82.12   | 120.46   | Extended context unhelpful — slight degradation |
| TF-01     | Transformer | d=64, h=4, L=2, ffn=128, lr=1e-4   | 151.87  | 191.93   | Under-capacity; LR too low |
| TF-02     | Transformer | d=128, h=4, L=3, ffn=256, lr=5e-4  | 109.99  | 154.80   | **Best Transformer — selected as final** |
| TF-03     | Transformer | dropout=0.2 at TF-02 config         | 127.36  | 178.64   | Higher dropout hurts — Exp 2 preferred |

---

## Final Model Comparison (Test Set — Dec 16–22, Square 5161)

| Model       | MAE   | MAPE (%) | RMSE   | Train Time (s) | Inference (ms/samp) |
|-------------|-------|----------|--------|----------------|---------------------|
| LSTM        | 85.15 | 8.84     | 128.29 | 980.5          | 0.5953              |
| TCN         | 80.87 | 8.88     | 118.23 | 1222.7         | 1.0272              |
| Transformer | 99.80 | 11.71    | 145.17 | 12364.1        | 2.3220              |

**Best model on Square 5161**: TCN (lowest MAE and RMSE).
**Best model on Squares 5059 and 5259**: LSTM (outperforms TCN by ~3–4 MAE units).
See Section 6 of the report for full discussion including Squares 5059 and 5259 results.
