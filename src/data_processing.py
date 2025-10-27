import os
import json
from glob import glob
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def load_csv_timeseries(path: str) -> pd.DataFrame:
    """Load CSV with columns time,open,high,low,close,volume into DataFrame with datetime index."""
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df = df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'})
    return df.sort_index()

def align_currencies(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align timestamps by union of indices and forward-fill missing values for each currency."""
    all_index = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    all_index = pd.DatetimeIndex(all_index)
    aligned = {}
    for name, df in dfs.items():
        reindexed = df.reindex(all_index)
        reindexed[['o','h','l','c','v']] = reindexed[['o','h','l','c','v']].ffill().bfill()
        aligned[name] = reindexed
    return aligned

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features: o, c, sma5, sma10, ema12, ema26, rsi14, macd, macd_signal"""
    out = pd.DataFrame(index=df.index)
    out['o'] = df['o'].astype(float)
    out['c'] = df['c'].astype(float)
    out['sma5'] = sma(out['c'], 5)
    out['sma10'] = sma(out['c'], 10)
    out['ema12'] = ema(out['c'], 12)
    out['ema26'] = ema(out['c'], 26)
    out['rsi14'] = rsi(out['c'], 14)
    macd_line, macd_signal = macd(out['c'] if 'c' in out else df['c'])
    out['macd'] = macd_line
    out['macd_sig'] = macd_signal
    out = out.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return out

def build_multi_currency_array(aligned_dfs: Dict[str, pd.DataFrame], feature_names: List[str]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Returns:
      data_arr: np.array shape (n_vars, n_time) where n_vars = n_currencies * n_features
      index: datetime index (common)
      var_names: list of variable names (currency.feature)
    """
    currencies = sorted(aligned_dfs.keys())
    n_time = len(next(iter(aligned_dfs.values())).index)
    var_list = []
    stacked = []
    for cur in currencies:
        df = aligned_dfs[cur]
        for f in feature_names:
            stacked.append(df[f].values.astype(float))
            var_list.append(f"{cur}.{f}")
    data_arr = np.vstack(stacked)
    return data_arr, next(iter(aligned_dfs.values())).index, var_list

def create_samples(data_arr: np.ndarray, seq_len: int, horizon: int = 2, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows across time dimension to create X and target indices."""
    n_vars, n_time = data_arr.shape
    samples, ys_idx = [], []
    if n_time < seq_len + horizon:
        return np.array([]), np.array([])
    for start in range(0, n_time - seq_len - horizon + 1, step):
        samples.append(data_arr[:, start:start+seq_len])
        ys_idx.append(start+seq_len)
    return np.stack(samples, axis=0), np.array(ys_idx, dtype=int)

def create_targets_for_currency(aligned_dfs: Dict[str, pd.DataFrame], currency_name: str, idx_array: np.ndarray, horizon: int = 2) -> np.ndarray:
    """Build target matrix y of shape (n_samples, horizon) containing future close prices."""
    df = aligned_dfs[currency_name]
    c_values = df['c'].values.astype(float)
    y = []
    for idx in idx_array:
        future = []
        for h in range(horizon):
            pos = idx + h
            future.append(c_values[pos] if pos < len(c_values) else c_values[-1])
        y.append(future)
    return np.array(y, dtype=float)

def normalize_per_currency(aligned_dfs: Dict[str, pd.DataFrame], feature_names: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    scalers, normalized = {}, {}
    for name, df in aligned_dfs.items():
        norm_df, stats = df.copy(), {}
        for f in feature_names:
            arr = df[f].values.astype(float)
            mu, sd = arr.mean(), arr.std()
            norm_df[f] = (arr - mu) / (sd if sd > 0 else 1.0)
            stats[f] = {'mu': float(mu), 'sd': float(sd if sd > 0 else 1.0)}
        normalized[name], scalers[name] = norm_df, stats
    return normalized, scalers

def load_and_process_data():
    from src.config import DATA_CSV_FOLDER, EVAL_CSV
    csv_paths = sorted(glob(os.path.join(DATA_CSV_FOLDER, '*.csv')))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {DATA_CSV_FOLDER}.")
    print(f"Found {len(csv_paths)} currency files for training.")

    train_dfs = {os.path.splitext(os.path.basename(p))[0]: load_csv_timeseries(p) for p in csv_paths if 'eval' not in p}

    if not os.path.exists(EVAL_CSV):
        raise RuntimeError(f"Evaluation CSV not found: {EVAL_CSV}")
    eval_name = os.path.splitext(os.path.basename(EVAL_CSV))[0]
    eval_df_raw = load_csv_timeseries(EVAL_CSV)
    print(f"Loaded evaluation currency: {eval_name}, length {len(eval_df_raw)}")

    combined = {**train_dfs, eval_name: eval_df_raw}
    aligned = align_currencies(combined)

    eval_df = aligned.pop(eval_name)
    train_aligned = aligned

    feature_names = ['o', 'c', 'sma5', 'sma10', 'ema12', 'ema26', 'rsi14', 'macd', 'macd_sig']
    train_features = {name: add_features(df) for name, df in train_aligned.items()}
    eval_features = add_features(eval_df)

    all_features = {**train_features, eval_name: eval_features}
    normalized, scalers = normalize_per_currency(all_features, feature_names)

    eval_norm = normalized.pop(eval_name)
    train_norm = normalized

    return train_norm, eval_norm, eval_name, scalers, all_features, feature_names