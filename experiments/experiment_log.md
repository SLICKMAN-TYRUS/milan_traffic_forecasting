# Experiment Log — Milan Traffic Forecasting

This document records the iterative hyperparameter tuning process for each model.
Each experiment documents: the parameters used, the results obtained, and the
reasoning behind subsequent changes.

---

## Area Under Investigation
- Primary area: Square [TOP_1] (highest total traffic — to be filled after EDA)
- Secondary areas: Square 4159, Square 4556

---

## Model 1: LSTM

### Experiment LSTM-01
**Rationale:** Baseline configuration from literature. Hidden size 64, 2 layers,
lookback of 1 day (144 steps at 10-min intervals) — common starting point in
cellular traffic forecasting (Huang et al. 2017).

| Parameter    | Value |
|-------------|-------|
| hidden_size  | 64    |
| num_layers   | 2     |
| dropout      | 0.2   |
| lookback     | 144   |
| lr           | 1e-3  |
| batch_size   | 64    |
| epochs       | 100   |

**Results (val set):**
| Metric | Value |
|--------|-------|
| MAE    | —     |
| MAPE   | —     |
| RMSE   | —     |

**Observations:** [Fill after running]

**Next steps:** [Fill after running — e.g., "Validation loss plateaued early;
try increasing hidden_size to 128 or extending lookback to 2 days to capture
the weekly pattern visible in ACF."]

---

### Experiment LSTM-02
**Rationale:** [Fill based on LSTM-01 results]

| Parameter    | Value |
|-------------|-------|
| hidden_size  | —     |
| num_layers   | —     |
| dropout      | —     |
| lookback     | —     |
| lr           | —     |
| batch_size   | —     |

**Results (val set):**
| Metric | Value |
|--------|-------|
| MAE    | —     |
| MAPE   | —     |
| RMSE   | —     |

**Observations:** [Fill after running]

**Next steps:** [Fill after running]

---

### Experiment LSTM-03 (Final)
**Rationale:** [Fill — summarise what the tuning found]

[Fill table and results]

---

## Model 2: TCN

### Experiment TCN-01
**Rationale:** Starting with the architecture from Bai et al. (2018): 4 residual
blocks with channels [32, 32, 64, 64], kernel size 3. This gives a receptive
field of 1 + 2×(3-1)×(1+2+4+8) = 61 timesteps ≈ 10 hours. EDA showed strong
autocorrelation up to at least 24h, so subsequent experiments will expand the
receptive field.

| Parameter    | Value             |
|-------------|-------------------|
| num_channels | [32, 32, 64, 64]  |
| kernel_size  | 3                 |
| dropout      | 0.2               |
| lookback     | 144               |
| lr           | 1e-3              |
| batch_size   | 64                |

**Results (val set):**
| Metric | Value |
|--------|-------|
| MAE    | —     |
| MAPE   | —     |
| RMSE   | —     |

**Next steps:** [Fill after running]

---

### Experiment TCN-02
[Fill]

---

### Experiment TCN-03 (Final)
[Fill]

---

## Model 3: Transformer

### Experiment TF-01
**Rationale:** Start with a small transformer (d_model=64, 2 heads, 2 layers)
to establish training stability before scaling. Transformers can be unstable
with high learning rates; starting at 1e-4 with pre-LN normalisation.

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

**Results (val set):**
| Metric | Value |
|--------|-------|
| MAE    | —     |
| MAPE   | —     |
| RMSE   | —     |

**Next steps:** [Fill after running]

---

### Experiment TF-02
[Fill]

---

### Experiment TF-03 (Final)
[Fill]

---

## Final Model Comparison (Test Set — Dec 16–22)

| Model       | MAE  | MAPE (%) | RMSE | Train Time (s) | Inference (ms/sample) |
|-------------|------|----------|------|----------------|-----------------------|
| LSTM        | —    | —        | —    | —              | —                     |
| TCN         | —    | —        | —    | —              | —                     |
| Transformer | —    | —        | —    | —              | —                     |

**Best model:** [Fill]
**Reasoning:** [Fill — quantitative + qualitative]
