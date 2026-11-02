"""Loading, cleaning and feature engineering for the India AQI dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
CORE_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

BUCKET_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_EDGES = [-np.inf, 50, 100, 200, 300, 400, np.inf]  # CPCB categories

NUM_FEATURES = (["AQI"] + CORE_POLLUTANTS +
                ["AQI_lag1", "AQI_lag2", "AQI_lag3", "AQI_roll7", "PM25_roll3",
                 "month_sin", "month_cos", "dayofweek"])
CAT_FEATURES = ["City"]
TARGET = "target"  # tomorrow's AQI

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
           5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}


def find_data_path():
    """Prefer the real Kaggle file, fall back to synthetic sample data."""
    for name in ("city_day.csv", "sample_city_day.csv"):
        p = ROOT / "data" / name
        if p.exists():
            return p
    return None


def bucket_series(values):
    """AQI values -> CPCB category names (as a string Series)."""
    s = pd.Series(np.atleast_1d(values), dtype=float)
    return pd.cut(s, BUCKET_EDGES, labels=BUCKET_LABELS).astype(str)


def bucket_of(value):
    return bucket_series([value]).iloc[0]


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.drop_duplicates(subset=["City", "Date"])


def clean(df):
    """Make each city a continuous daily series and fill short gaps.

    - reindex to every calendar day (so lag features really mean 'yesterday')
    - time-based interpolation, max 5 consecutive days (longer gaps stay missing)
    - rebuild AQI_Bucket from AQI so the labels are consistent
    """
    num_cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    parts = []
    for city, g in df.groupby("City"):
        g = g.drop(columns="City").set_index("Date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g.index.name = "Date"
        g[num_cols] = g[num_cols].interpolate(method="time", limit=5,
                                              limit_direction="both")
        g.insert(0, "City", city)
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["AQI_Bucket"] = bucket_series(out["AQI"].values)
    out.loc[out["AQI"].isna(), "AQI_Bucket"] = np.nan
    return out


def build_features(df):
    """Add lag / rolling / calendar features and the next-day target."""
    parts = []
    for _, g in df.groupby("City"):
        g = g.sort_values("Date").copy()
        for k in (1, 2, 3):
            g[f"AQI_lag{k}"] = g["AQI"].shift(k)
        g["AQI_roll7"] = g["AQI"].rolling(7, min_periods=4).mean()
        g["PM25_roll3"] = g["PM2.5"].rolling(3, min_periods=2).mean()
        g[TARGET] = g["AQI"].shift(-1)
        parts.append(g)
    f = pd.concat(parts, ignore_index=True)
    f["month"] = f["Date"].dt.month
    f["season"] = f["month"].map(SEASONS)
    f["dayofweek"] = f["Date"].dt.dayofweek
    f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)
    return f


def training_frame(feats):
    """Rows where both today's AQI and tomorrow's AQI are known."""
    return feats.dropna(subset=["AQI", TARGET]).reset_index(drop=True)


def time_split(frame, test_frac=0.2):
    """Chronological split (no shuffling -> no leakage from the future)."""
    dates = np.sort(frame["Date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    return (frame[frame["Date"] < cutoff], frame[frame["Date"] >= cutoff],
            pd.Timestamp(cutoff))
