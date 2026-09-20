"""
eda.py
======
Exploratory Data Analysis for the Milan Internet traffic dataset.

Covers all analyses required by Section 2 of the assignment:
  1. Distribution of total traffic across 10,000 geographic areas
  2. Identify top-3 areas by total traffic
  3. Time-series plots for top-3, square 4159, and square 4556 (first 2 weeks)
  4. Two additional analyses on the highest-traffic area:
       a. Autocorrelation / partial autocorrelation (ACF/PACF)
       b. Seasonal decomposition (weekly + daily seasonality)
       c. (Optional extras: ADF stationarity test, anomaly detection)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})
PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# 1. Total traffic distribution
# ---------------------------------------------------------------------------

def plot_traffic_distribution(matrix: pd.DataFrame, save_dir: str = None) -> pd.Series:
    """
    Plot distribution of total Internet traffic across all 10,000 geographic areas.

    Parameters
    ----------
    matrix : pd.DataFrame  (T × 10000) traffic matrix
    save_dir : str or None  directory to save figure

    Returns
    -------
    pd.Series : total traffic per square_id (sorted descending)
    """
    total_traffic = matrix.sum(axis=0).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(total_traffic.values, bins=100, color=PALETTE[0], edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Total Internet Traffic (sum over observation period)")
    axes[0].set_ylabel("Number of Geographic Areas")
    axes[0].set_title("Distribution of Total Internet Traffic\nacross 10,000 Geographic Areas")
    axes[0].axvline(total_traffic.median(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Median = {total_traffic.median():.1f}")
    axes[0].axvline(total_traffic.mean(), color="orange", linestyle="--", linewidth=1.2,
                    label=f"Mean = {total_traffic.mean():.1f}")
    axes[0].legend()

    # Log-scale version to reveal the tail
    axes[1].hist(total_traffic.values, bins=100, color=PALETTE[1], edgecolor="white", linewidth=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Total Internet Traffic")
    axes[1].set_ylabel("Number of Areas (log scale)")
    axes[1].set_title("Distribution (Log Scale)")

    fig.suptitle("Total Internet Traffic Distribution — Milan Dataset", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "fig1_traffic_distribution.png"), bbox_inches="tight")
        print("Saved: fig1_traffic_distribution.png")

    plt.show()

    # Summary stats
    print("\nTraffic Distribution Summary")
    print("-" * 40)
    print(f"  Min    : {total_traffic.min():.2f}")
    print(f"  25th % : {total_traffic.quantile(0.25):.2f}")
    print(f"  Median : {total_traffic.median():.2f}")
    print(f"  Mean   : {total_traffic.mean():.2f}")
    print(f"  75th % : {total_traffic.quantile(0.75):.2f}")
    print(f"  Max    : {total_traffic.max():.2f}")
    print(f"  Std    : {total_traffic.std():.2f}")
    print(f"  Skewness : {total_traffic.skew():.3f}")

    return total_traffic


# ---------------------------------------------------------------------------
# 2. Identify top-N areas
# ---------------------------------------------------------------------------

def get_top_areas(total_traffic: pd.Series, n: int = 3) -> list:
    """Return list of square_ids with highest total traffic."""
    top = total_traffic.nlargest(n).index.tolist()
    print(f"\nTop {n} areas by total Internet traffic:")
    for rank, sq in enumerate(top, 1):
        print(f"  #{rank}: Square {sq}  —  total traffic = {total_traffic[sq]:.2f}")
    return top


# ---------------------------------------------------------------------------
# 3. Time series plots (first two weeks)
# ---------------------------------------------------------------------------

def plot_time_series(
    matrix: pd.DataFrame,
    square_ids: list,
    save_dir: str = None,
    n_weeks: int = 2,
    labels: dict = None,
) -> None:
    """
    Plot Internet traffic time series for specified areas over first n_weeks.

    Parameters
    ----------
    matrix     : (T × 10000) traffic matrix with DatetimeIndex
    square_ids : list of square IDs to plot
    save_dir   : directory to save figure
    n_weeks    : number of weeks to plot from start
    labels     : optional dict {square_id: label_str} for legend
    """
    cutoff = matrix.index[0] + pd.Timedelta(weeks=n_weeks)
    subset = matrix.loc[matrix.index <= cutoff, square_ids]

    n = len(square_ids)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sq_id in zip(axes, square_ids):
        label = (labels or {}).get(sq_id, f"Square {sq_id}")
        ax.plot(subset.index, subset[sq_id], linewidth=0.7, color=PALETTE[square_ids.index(sq_id) % 10])
        ax.set_ylabel("Internet Traffic", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")

        # Shade weekends
        _shade_weekends(ax, subset.index)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(f"Internet Traffic Time Series — First {n_weeks} Weeks", fontsize=13)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = f"fig2_time_series_{'_'.join(str(s) for s in square_ids)}.png"
        fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
        print(f"Saved: {fname}")

    plt.show()


def _shade_weekends(ax, index: pd.DatetimeIndex) -> None:
    """Add light grey shading to weekend days on an axis."""
    dates = pd.Series(index).dt.date.unique()
    for d in dates:
        ts = pd.Timestamp(d, tz="UTC")
        if ts.weekday() >= 5:          # Sat=5, Sun=6
            ax.axvspan(ts, ts + pd.Timedelta(days=1), alpha=0.08, color="grey", zorder=0)


# ---------------------------------------------------------------------------
# 4a. Autocorrelation / PACF analysis
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    series: pd.Series,
    nlags: int = 144 * 2,      # 2 days at 10-min intervals (144 per day)
    save_dir: str = None,
    title_prefix: str = "",
) -> dict:
    """
    Compute and plot ACF and PACF for a traffic time series.

    Parameters
    ----------
    series     : pd.Series — single area traffic time series
    nlags      : number of lags to compute
    save_dir   : directory to save figure
    title_prefix : prefix for plot title

    Returns
    -------
    dict with keys 'acf_values', 'pacf_values'
    """
    # Drop any NaN
    s = series.dropna().values.astype(float)

    acf_vals  = acf(s,  nlags=nlags, fft=True)
    pacf_vals = pacf(s, nlags=min(nlags, len(s) // 2 - 1))

    # Confidence interval (approximate, 95%)
    conf = 1.96 / np.sqrt(len(s))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # ACF
    lags = np.arange(len(acf_vals))
    axes[0].bar(lags, acf_vals, width=0.8, color=PALETTE[0], alpha=0.7)
    axes[0].axhline(conf,  color="red", linestyle="--", linewidth=0.8, label="95% CI")
    axes[0].axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Lag (× 10 min)")
    axes[0].set_ylabel("ACF")
    axes[0].set_title(f"{title_prefix} Autocorrelation Function (ACF)")
    axes[0].legend()

    # Add lag-time annotations for key periodicities
    for lag, label in [(144, "1 day"), (288, "2 days"), (1008, "1 week")]:
        if lag <= nlags:
            axes[0].axvline(lag, color="green", linestyle=":", linewidth=1, alpha=0.7)
            axes[0].text(lag + 2, axes[0].get_ylim()[1] * 0.9, label, fontsize=7, color="green")

    # PACF
    plags = np.arange(len(pacf_vals))
    axes[1].bar(plags, pacf_vals, width=0.8, color=PALETTE[2], alpha=0.7)
    axes[1].axhline(conf,  color="red", linestyle="--", linewidth=0.8, label="95% CI")
    axes[1].axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Lag (× 10 min)")
    axes[1].set_ylabel("PACF")
    axes[1].set_title(f"{title_prefix} Partial Autocorrelation Function (PACF)")
    axes[1].legend()

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "fig3_acf_pacf.png"), bbox_inches="tight")
        print("Saved: fig3_acf_pacf.png")

    plt.show()

    return {"acf_values": acf_vals, "pacf_values": pacf_vals}


# ---------------------------------------------------------------------------
# 4b. Seasonal decomposition
# ---------------------------------------------------------------------------

def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 144,          # 1 day = 144 × 10-min intervals
    save_dir: str = None,
    title_prefix: str = "",
) -> None:
    """
    Additive seasonal decomposition of a traffic time series.

    Parameters
    ----------
    series   : pd.Series — single area traffic time series
    period   : seasonal period in number of observations
    save_dir : directory to save figure
    """
    # Use first 4 weeks for clarity
    s = series.dropna().iloc[:144 * 7 * 4]

    decomp = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (s,                "Observed"),
        (decomp.trend,     "Trend"),
        (decomp.seasonal,  "Seasonal"),
        (decomp.resid,     "Residual"),
    ]
    for ax, (data, label) in zip(axes, components):
        ax.plot(data.index, data.values, linewidth=0.7, color=PALETTE[0])
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(f"{title_prefix} Seasonal Decomposition (period = {period} × 10 min = 1 day)",
                 fontsize=12)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "fig4_seasonal_decomp.png"), bbox_inches="tight")
        print("Saved: fig4_seasonal_decomp.png")

    plt.show()


# ---------------------------------------------------------------------------
# 4c. Stationarity test (ADF)
# ---------------------------------------------------------------------------

def adf_stationarity_test(series: pd.Series, title: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns dict with test statistic, p-value, and verdict.
    """
    s = series.dropna().values.astype(float)
    result = adfuller(s, autolag="AIC")

    out = {
        "adf_statistic": result[0],
        "p_value":        result[1],
        "n_lags":         result[2],
        "n_obs":          result[3],
        "critical_values": result[4],
        "stationary":     result[1] < 0.05,
    }

    print(f"\nADF Stationarity Test — {title}")
    print("-" * 45)
    print(f"  ADF Statistic : {out['adf_statistic']:.4f}")
    print(f"  p-value        : {out['p_value']:.6f}")
    for cv_label, cv_val in out["critical_values"].items():
        print(f"  Critical ({cv_label}) : {cv_val:.4f}")
    verdict = "STATIONARY" if out["stationary"] else "NON-STATIONARY"
    print(f"  Verdict        : {verdict} (α=0.05)")

    return out


# ---------------------------------------------------------------------------
# 4d. Intraday / intraweek average pattern
# ---------------------------------------------------------------------------

def plot_average_daily_weekly_pattern(
    series: pd.Series,
    save_dir: str = None,
    title_prefix: str = "",
) -> None:
    """
    Plot average intraday (hourly) and intraweek (day-of-week) traffic patterns.
    Useful for understanding periodic structure before modelling.
    """
    df = series.to_frame(name="traffic")
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["interval"]   = df.index.hour * 6 + df.index.minute // 10  # 0..143

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average by 10-min interval within day
    daily_avg = df.groupby("interval")["traffic"].mean()
    x_hours = np.arange(len(daily_avg)) / 6               # convert to hours
    axes[0].plot(x_hours, daily_avg.values, color=PALETTE[0], linewidth=1.5)
    axes[0].fill_between(
        x_hours,
        df.groupby("interval")["traffic"].quantile(0.25).values,
        df.groupby("interval")["traffic"].quantile(0.75).values,
        alpha=0.2, color=PALETTE[0], label="IQR (25th–75th %ile)"
    )
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Average Internet Traffic")
    axes[0].set_title(f"{title_prefix}\nAverage Intraday Pattern")
    axes[0].set_xticks(range(0, 25, 3))
    axes[0].legend()

    # Average by day of week
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_avg = df.groupby("dayofweek")["traffic"].mean()
    axes[1].bar(
        range(7), weekly_avg.values,
        color=[PALETTE[0] if d < 5 else PALETTE[3] for d in range(7)],
        edgecolor="white"
    )
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_labels)
    axes[1].set_ylabel("Average Internet Traffic")
    axes[1].set_title(f"{title_prefix}\nAverage by Day of Week")

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "fig5_daily_weekly_pattern.png"), bbox_inches="tight")
        print("Saved: fig5_daily_weekly_pattern.png")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_full_eda(matrix: pd.DataFrame, save_dir: str = "report/figures") -> dict:
    """
    Run the complete EDA pipeline and return key results.

    Parameters
    ----------
    matrix   : (T × 10000) processed traffic matrix
    save_dir : directory to save all figures
    """
    results = {}

    # 1. Distribution
    print("=" * 60)
    print("STEP 1: Traffic Distribution")
    print("=" * 60)
    total_traffic = plot_traffic_distribution(matrix, save_dir=save_dir)
    results["total_traffic"] = total_traffic

    # 2. Top areas
    top3 = get_top_areas(total_traffic, n=3)
    results["top3"] = top3
    highest = top3[0]

    # 3. Time series plots
    print("\n" + "=" * 60)
    print("STEP 2: Time Series Plots")
    print("=" * 60)

    special_ids = [4159, 4556]
    all_ids     = top3 + [sid for sid in special_ids if sid not in top3]
    labels      = {sid: f"Square {sid} (Top {i+1} traffic)" for i, sid in enumerate(top3)}
    labels[4159] = "Square 4159 (specified)"
    labels[4556] = "Square 4556 (specified)"

    plot_time_series(matrix, all_ids, save_dir=save_dir, n_weeks=2, labels=labels)

    # 4. Additional analyses on highest-traffic area
    print("\n" + "=" * 60)
    print(f"STEP 3: Additional Analyses — Square {highest}")
    print("=" * 60)

    series_highest = matrix[highest]

    # 4a. ACF/PACF
    acf_results = plot_acf_pacf(
        series_highest, nlags=288,   # 2 days
        save_dir=save_dir,
        title_prefix=f"Square {highest}"
    )
    results["acf_pacf"] = acf_results

    # 4b. Seasonal decomposition
    plot_seasonal_decomposition(
        series_highest, period=144,  # 1 day
        save_dir=save_dir,
        title_prefix=f"Square {highest}"
    )

    # 4c. Stationarity
    adf_result = adf_stationarity_test(series_highest, title=f"Square {highest}")
    results["adf"] = adf_result

    # 4d. Intraday / weekly patterns
    plot_average_daily_weekly_pattern(
        series_highest,
        save_dir=save_dir,
        title_prefix=f"Square {highest}"
    )

    return results


if __name__ == "__main__":
    import argparse
    from data_loader import load_processed

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--output_dir",    default="report/figures")
    args = parser.parse_args()

    matrix = load_processed(os.path.join(args.processed_dir, "traffic_matrix.parquet"))
    run_full_eda(matrix, save_dir=args.output_dir)
