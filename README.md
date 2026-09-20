# Milan Mobile Network Traffic Forecasting

Comparative analysis of sequential models (LSTM, TCN, Transformer) for one-step-ahead
mobile network traffic forecasting on the Telecom Italia Big Data Challenge dataset.

## Project Structure

```
milan-traffic-forecasting/
├── data/                        # Raw .txt files go here (not committed to git)
├── notebooks/
│   ├── 01_data_loading_memory.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_model_selection.ipynb
│   └── 04_forecasting_experiments.ipynb
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
│   └── experiment_log.md        # Iterative tuning decisions & results
├── report/
│   └── figures/                 # Saved plots
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/milan-traffic-forecasting.git
cd milan-traffic-forecasting
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the data
Download the Telecom Italia Big Data Challenge dataset from:
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

Place all daily `.txt` files in the `data/` directory. Files should be named in the pattern:
`sms-call-internet-mi-YYYY-MM-DD.txt`

## Running the Project

### Option A: Notebooks (recommended for exploration)
Launch Jupyter and run notebooks in order (01 → 04):
```bash
jupyter lab
```

### Option B: Scripts (recommended for full pipeline)
```bash
# Step 1: Load and preprocess all data
python src/data_loader.py --data_dir data/ --output_dir data/processed/

# Step 2: Run EDA and save figures
python src/eda.py --processed_dir data/processed/ --output_dir report/figures/

# Step 3: Train all models and run experiments
python src/train.py --processed_dir data/processed/ --output_dir experiments/

# Step 4: Evaluate and generate result tables + plots
python src/evaluate.py --experiments_dir experiments/ --output_dir report/figures/
```

## Data Format

Each raw `.txt` file is tab-separated with **no header**, containing columns:
| Index | Name          | Description                              |
|-------|---------------|------------------------------------------|
| 0     | square_id     | Geographic cell ID (1–10,000)            |
| 1     | time_interval | Unix timestamp in milliseconds           |
| 2     | country_code  | Country code of the activity             |
| 3     | sms_in        | Incoming SMS activity (normalized)       |
| 4     | sms_out       | Outgoing SMS activity (normalized)       |
| 5     | call_in       | Incoming call activity (normalized)      |
| 6     | call_out      | Outgoing call activity (normalized)      |
| 7     | internet      | Internet traffic activity (normalized)   |

The pipeline aggregates internet traffic per `(square_id, time_interval)` across all country codes,
producing a clean matrix of shape `(T, 10000)` where T is the number of 10-minute intervals.

## Models

| Model       | Architecture          | Key Strength                            |
|-------------|----------------------|-----------------------------------------|
| LSTM        | Stacked LSTM (2-layer) | Captures sequential dependencies        |
| TCN         | Dilated Causal Conv   | Long receptive field, parallelizable   |
| Transformer | Multi-head Attention  | Non-local pattern capture via attention |

## Hardware & Reproducibility

All experiments were run on [your hardware here]. Random seeds are fixed to 42 across
PyTorch, NumPy, and Python for reproducibility.

## References

- Barlacchi et al. (2015). A multi-source dataset of urban life in the city of Milan.
  *Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55
- Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
