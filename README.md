# Milan Mobile Network Traffic Forecasting

Comparative analysis of sequential models (LSTM, TCN, Transformer) for one-step-ahead
mobile network traffic forecasting on the Telecom Italia Big Data Challenge dataset.

## Project Structure

```
milan-traffic-forecasting/
├── data/                        # Raw .txt files go here (not committed to git)
│   └── processed/               # Processed matrix saved here after NB01
├── notebooks/
│   ├── 01_data_loading_memory.ipynb   # Memory-efficient loading & evidence
│   ├── 02_exploratory_analysis.ipynb  # EDA: distribution, time series, ACF, decomp
│   ├── 03_model_selection.ipynb       # Model justification & parameter counts
│   └── 04_forecasting_experiments.ipynb  # Training, tuning, evaluation
├── src/
│   ├── data_loader.py           # Memory-efficient data loading & aggregation
│   ├── eda.py                   # EDA helper functions
│   ├── features.py              # Sequence building, normalization
│   ├── models/
│   │   ├── lstm_model.py        # Stacked LSTM
│   │   ├── tcn_model.py         # Temporal Convolutional Network
│   │   └── transformer_model.py # Time-series Transformer
│   ├── train.py                 # Training loop with timing & experiment logging
│   ├── evaluate.py              # MAE, MAPE, RMSE, plots
│   └── tuning.py                # Hyperparameter search utilities
├── experiments/
│   ├── experiment_log.jsonl     # Machine-readable iterative tuning log
│   └── experiment_log.md        # Human-readable tuning decisions & results
├── report/
│   └── figures/                 # All saved plots (EDA + predictions + failure cases)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/SLICKMAN-TYRUS/milan_traffic_forecasting.git
cd milan_traffic_forecasting
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows Command Prompt
.\venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `fastparquet` and `pyarrow` are both required for saving and loading
> the processed traffic matrix. Both are listed in `requirements.txt`.

### 4. Add the data
Download the Telecom Italia Big Data Challenge dataset from:
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

Place all 62 daily `.txt` files directly in the `data/` directory. Files must follow
the naming pattern: `sms-call-internet-mi-YYYY-MM-DD.txt`

If the data files are stored elsewhere on your machine, update `DATA_DIR` in
Notebook 01 to point to their location. Example:
```python
DATA_DIR = r'C:\Users\YourName\milan_data'   # Windows absolute path
```

## Running the Project

Run notebooks **in order** (01 → 02 → 03 → 04). Each notebook depends on outputs
from the previous one.

```bash
jupyter lab
# or open notebooks directly in VS Code
```

### Notebook 01 — Data Loading & Memory Management
Loads all 62 raw files, aggregates internet traffic per cell per timestamp,
and saves `data/processed/traffic_matrix.parquet` (~343 MB).
**Runtime: 20–60 minutes depending on hardware.**

> **Important note on column types:** After loading the processed matrix,
> column names (square IDs) must be cast to integers:
> ```python
> matrix.columns = matrix.columns.astype(int)
> ```
> This is handled automatically in all notebooks but is required if you load
> the Parquet file manually.

### Notebook 02 — Exploratory Data Analysis
Generates all EDA figures saved to `report/figures/`. Identifies the top-3
geographic areas and saves `data/processed/top3_areas.json`.

### Notebook 03 — Model Selection
Documents model architecture justification and computes parameter counts.
No data required — runs in under 1 minute.

### Notebook 04 — Forecasting Experiments
Runs all hyperparameter tuning experiments (9 total) and final evaluations
(9 models × 3 areas). **Runtime: 4–8 hours on CPU.**
All results saved to `experiments/` as JSON files.

## Data Format

Each raw `.txt` file is tab-separated with **no header**, containing 8 columns:

| Index | Name          | Description                              |
|-------|---------------|------------------------------------------|
| 0     | square_id     | Geographic cell ID (1–10,000)            |
| 1     | time_interval | Unix timestamp in milliseconds           |
| 2     | country_code  | Country code of the activity             |
| 3     | sms_in        | Incoming SMS activity                    |
| 4     | sms_out       | Outgoing SMS activity                    |
| 5     | call_in       | Incoming call activity                   |
| 6     | call_out      | Outgoing call activity                   |
| 7     | internet      | **Internet traffic activity** ← target   |

The pipeline aggregates internet traffic per `(square_id, time_interval)` across
all country codes, producing a matrix of shape `(8928, 10000)` — 62 days × 144
10-minute intervals per day, across 10,000 geographic cells.

## Models

| Model       | Architecture            | Params  | Key Strength                          |
|-------------|------------------------|---------|---------------------------------------|
| LSTM        | Stacked LSTM (2-layer)  | 52,545  | Gated sequential hidden state         |
| TCN         | Dilated Causal Conv     | 144,897 | Long receptive field, parallelisable  |
| Transformer | Multi-head Attention    | 406,017 | Global pairwise dependency modelling  |

## Hardware & Reproducibility

All experiments were run on **Windows 11 (Intel CPU, PyTorch 2.14.0, CPU-only)**.
Random seeds are fixed to **42** across PyTorch, NumPy, and Python for full
reproducibility. Expected total training time on similar hardware: ~12–16 hours.

## Key Results (December 16–22, 2013 test period)

| Model       | Square 5161 MAE | Square 5059 MAE | Square 5259 MAE |
|-------------|----------------|----------------|----------------|
| LSTM        | 85.15          | **71.25**      | **67.86**      |
| TCN         | **80.87**      | 74.70          | 69.44          |
| Transformer | 99.80          | 101.95         | 76.76          |

TCN achieves the best performance on the highest-traffic area (Square 5161).
LSTM wins on Squares 5059 and 5259. Transformer consistently underperforms both.

## References

- Barlacchi et al. (2015). A multi-source dataset of urban life in the city of Milan.
  *Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55
- Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- Bai et al. (2018). An empirical evaluation of generic convolutional and recurrent
  networks for sequence modeling. arXiv:1803.01271.
