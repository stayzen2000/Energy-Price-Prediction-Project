from __future__ import annotations
import pandas as pd

def make_time_splits(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Returns three dataframes: train, val, test based on timestamp cutoffs.
    Assumes df has a 'timestamp' column (timezone-aware) and is sorted.
    """
    train_end_ts = pd.to_datetime(train_end, utc=True)
    val_end_ts = pd.to_datetime(val_end, utc=True)

    train = df[df["timestamp"] <= train_end_ts].copy()
    val = df[(df["timestamp"] > train_end_ts) & (df["timestamp"] <= val_end_ts)].copy()
    test = df[df["timestamp"] > val_end_ts].copy()

    return train, val, test
