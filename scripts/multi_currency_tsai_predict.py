"""
multi_currency_tsai_predict.py

Requirements:
  pip install tsai fastai pandas numpy matplotlib scikit-learn
(If your environment lacks tsai, install it. This script avoids deprecated tsai functions.)

Usage:
  - Put one JSON file per currency in ./data_json/ (configurable)
  - Pick one filename as the evaluation (unseen) currency or place eval file in ./eval_json/
  - Run the script: python multi_currency_tsai_predict.py
"""

import os
import json
from glob import glob
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# tsai & fastai imports
from tsai.all import get_ts_dls, TSStandardize
from fastai.learner import Learner
from fastai.callback.all import SaveModelCallback

# -------------------------
# Config
# -------------------------
DATA_FOLDER = './data_json'   # JSON files for training currencies
EVAL_JSON = './eval_json/eval_currency.json'  # JSON file for evaluation/unseen currency
MODEL_SAVE_PATH = './trained_model.pth'
CANDIDATE_SEQ_LENS = [8, 16, 24]  # candidate lookback lengths to pick from (keeps search reasonable)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------
# Utility: technical indicators
# -------------------------
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

# -------------------------
# Load single JSON -> DataFrame
# -------------------------
def load_json_timeseries(path: str) -> pd.DataFrame:
    """Load JSON with keys t,o,h,l,c,v into DataFrame with datetime index."""
    with open(path, 'r') as f:
        j = json.load(f)
    # Expect timestamps in j['t']
    df = pd.DataFrame({
        't': pd.to_datetime(j['t'], unit='s', utc=True) if isinstance(j['t'][0], (int, float)) else pd.to_datetime(j['t']),
        'o': j['o'],
        'h': j.get('h', [np.nan]*len(j['t'])),
        'l': j.get('l', [np.nan]*len(j['t'])),
        'c': j['c'],
        'v': j.get('v', [np.nan]*len(j['t'])),
    })
    df = df.set_index('t').sort_index()
    return df

# -------------------------
# Align timestamps across currencies (union) and forward-fill
# -------------------------
def align_currencies(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align timestamps by union of indices and forward-fill missing values for each currency."""
    # union of all timestamps
    all_index = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    all_index = pd.DatetimeIndex(all_index)
    aligned = {}
    for name, df in dfs.items():
        reindexed = df.reindex(all_index)
        # forward fill then back fill if needed (so earliest rows also get values)
        reindexed[['o','h','l','c','v']] = reindexed[['o','h','l','c','v']].ffill().bfill()
        aligned[name] = reindexed
    return aligned

# -------------------------
# Feature engineering per currency (use only o and c plus indicators)
# -------------------------
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
    # fill any remaining NaNs
    out = out.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return out

# -------------------------
# Build stacked multi-currency dataset
# -------------------------
def build_multi_currency_array(aligned_dfs: Dict[str, pd.DataFrame], feature_names: List[str]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Returns:
      data_arr: np.array shape (n_vars, n_time) where n_vars = n_currencies * n_features
      index: datetime index (common)
      var_names: list of variable names (currency.feature)
    tsai expects shape (n_samples, n_vars, seq_len) for get_ts_dls -- we'll produce samples later.
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
    data_arr = np.vstack(stacked)  # shape (n_vars, n_time)
    return data_arr, next(iter(aligned_dfs.values())).index, var_list

# -------------------------
# Sequence builder
# -------------------------
def create_samples(data_arr: np.ndarray, seq_len: int, horizon: int = 2, step: int = 1) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Slide windows across time dimension to create X, y.
    data_arr shape: (n_vars, n_time)
    X shape: (n_samples, n_vars, seq_len)
    y shape: (n_samples, horizon) -> actual next closes for the *evaluation currency only*, but we will train to predict global next closes by taking last currency's close as target?
    Approach: We will build targets as the next two close values of a selected "target currency index" (we will set target_index later).
    For training, we'll train to predict the aggregate next closes using the last currency in var grouping as a "global target proxy". However better: we will include per-sample targets passed externally. So this function just creates generic windows; targets are created outside.
    """
    n_vars, n_time = data_arr.shape
    samples = []
    ys_idx = []
    if n_time < seq_len + horizon:
        return np.array([]), np.array([])
    for start in range(0, n_time - seq_len - horizon + 1, step):
        samples.append(data_arr[:, start:start+seq_len])
        ys_idx.append(start+seq_len)  # index of first forecast point
    X = np.stack(samples, axis=0)  # (n_samples, n_vars, seq_len)
    return X, np.array(ys_idx, dtype=int)

# -------------------------
# Create target vectors for a specific currency
# -------------------------
def create_targets_for_currency(aligned_dfs: Dict[str, pd.DataFrame], currency_name: str, idx_array: np.ndarray, horizon: int = 2) -> np.ndarray:
    """
    Build target matrix y of shape (n_samples, horizon) containing future close prices for currency_name
    idx_array contains the integer time indices corresponding to the first forecast time.
    """
    df = aligned_dfs[currency_name]
    c_values = df['c'].values.astype(float)
    y = []
    for idx in idx_array:
        future = []
        for h in range(horizon):
            pos = idx + h
            if pos < len(c_values):
                future.append(c_values[pos])
            else:
                future.append(c_values[-1])
        y.append(future)
    return np.array(y, dtype=float)  # shape (n_samples, horizon)

# -------------------------
# Normalize each currency individually (per-currency per-feature)
# -------------------------
def normalize_per_currency(aligned_dfs: Dict[str, pd.DataFrame], feature_names: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    scalers = {}
    normalized = {}
    for name, df in aligned_dfs.items():
        norm_df = df.copy()
        stats = {}
        for f in feature_names:
            arr = df[f].values.astype(float)
            mu = arr.mean()
            sd = arr.std() if arr.std() > 0 else 1.0
            norm_df[f] = (arr - mu) / sd
            stats[f] = {'mu': float(mu), 'sd': float(sd)}
        normalized[name] = norm_df
        scalers[name] = stats
    return normalized, scalers

# -------------------------
# Custom PyTorch model: parallel CNN + LSTM -> concat -> heads
# -------------------------
class CNNBranch(nn.Module):
    def __init__(self, in_channels, seq_len, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(F.relu(self.fc(x)))
        return x  # (batch, hidden_dim)

class LSTMBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, n_layers=1):
        super().__init__()
        # We'll treat variables as input_size; transpose before LSTM
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        # x: (batch, channels, seq_len)
        # transpose to (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        # take last time step
        last = out[:, -1, :]
        last = self.dropout(F.relu(self.fc(last)))
        return last  # (batch, hidden_dim)

class ParallelCNNLSTM(nn.Module):
    def __init__(self, n_vars, seq_len, hidden_dim=128, horizon=2):
        super().__init__()
        self.cnn = CNNBranch(in_channels=n_vars, seq_len=seq_len, hidden_dim=hidden_dim)
        self.lstm = LSTMBranch(in_channels=n_vars, hidden_dim=hidden_dim)
        # combined
        self.combined_fc = nn.Linear(hidden_dim*2, hidden_dim)
        # heads: regression for horizon values, and classification (sigmoid) for upward probability for each horizon
        self.reg_head = nn.Linear(hidden_dim, horizon)   # output predicted close values (unnormalized)
        self.cls_head = nn.Linear(hidden_dim, horizon)   # output logits for upward-probability
    def forward(self, x):
        # x: (batch, n_vars, seq_len)
        c = self.cnn(x)
        l = self.lstm(x)
        stacked = torch.cat([c, l], dim=1)
        h = F.relu(self.combined_fc(stacked))
        preds = self.reg_head(h)
        logits = self.cls_head(h)
        probs = torch.sigmoid(logits)
        return preds, probs  # preds: (batch, horizon), probs: (batch, horizon)

# -------------------------
# Combined loss: MSE for regression + BCE for classification (probabilities)
# -------------------------
def combined_loss(preds: torch.Tensor, probs: torch.Tensor, y_reg: torch.Tensor, y_cls: torch.Tensor, alpha=1.0, beta=0.5):
    """
    y_reg: (batch, horizon) true future closes (normalized)
    y_cls: (batch, horizon) binary labels: 1 if future close > last observed close, else 0
    alpha: weight for regression MSE
    beta: weight for classification BCE
    """
    mse = F.mse_loss(preds, y_reg)
    bce = F.binary_cross_entropy(probs, y_cls)
    return alpha * mse + beta * bce

# -------------------------
# Metrics: MAE, RMSE, MAPE (computed in original scale)
# -------------------------
def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape_np(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# -------------------------
# Main flow
# -------------------------
def load_and_process_data():
    json_paths = sorted(glob(os.path.join(DATA_FOLDER, '*.json')))
    if not json_paths:
        raise RuntimeError(f"No JSON files found in {DATA_FOLDER}.")
    print(f"Found {len(json_paths)} currency files for training.")

    train_dfs = {os.path.splitext(os.path.basename(p))[0]: load_json_timeseries(p) for p in json_paths}

    if not os.path.exists(EVAL_JSON):
        raise RuntimeError(f"Evaluation JSON not found: {EVAL_JSON}")
    eval_name = os.path.splitext(os.path.basename(EVAL_JSON))[0]
    eval_df_raw = load_json_timeseries(EVAL_JSON)
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

def find_best_sequence_length(data_arr):
    n_time = data_arr.shape[1]
    for seq_len in sorted(CANDIDATE_SEQ_LENS, reverse=True):
        if n_time > seq_len + 2: # horizon
            print(f"Selected best_seq = {seq_len}")
            return seq_len
    raise RuntimeError("Not enough data for any candidate sequence length.")

def train_model(X_train, y_train_reg, y_train_cls, X_valid, y_valid_reg, y_valid_cls, n_vars, seq_len):
    Xtr, Xva = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_valid, dtype=torch.float32)
    ytr_reg, yva_reg = torch.tensor(y_train_reg, dtype=torch.float32), torch.tensor(y_valid_reg, dtype=torch.float32)
    ytr_cls, yva_cls = torch.tensor(y_train_cls, dtype=torch.float32), torch.tensor(y_valid_cls, dtype=torch.float32)

    train_ds, valid_ds = TensorDataset(Xtr, ytr_reg, ytr_cls), TensorDataset(Xva, yva_reg, yva_cls)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE)

    model = ParallelCNNLSTM(n_vars=n_vars, seq_len=seq_len, hidden_dim=128, horizon=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_state, best_epoch_val = None, float('inf')

    print("Training final model...")
    for epoch in range(EPOCHS):
        model.train()
        train_losses = [combined_loss(model(xb)[0], model(xb)[1], yregb, yclsb) for xb, yregb, yclsb in train_loader]
        for loss in train_losses:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_losses = [combined_loss(model(xb)[0], model(xb)[1], yregb, yclsb) for xb, yregb, yclsb in valid_loader]

        train_loss, val_loss = np.mean([l.item() for l in train_losses]), np.mean([l.item() for l in val_losses])
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_epoch_val:
            best_epoch_val, best_model_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_model_state

def evaluate_and_plot(model_state, scalers, feature_names, seq_len, var_names, X_all, idx_array, eval_name, all_features, index):
    n_vars = X_all.shape[1]
    model = ParallelCNNLSTM(n_vars=n_vars, seq_len=seq_len, hidden_dim=128, horizon=2)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    X_tensor_all = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    preds_list, probs_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor_all), BATCH_SIZE):
            preds_b, probs_b = model(X_tensor_all[i:i+BATCH_SIZE])
            preds_list.append(preds_b.cpu().numpy())
            probs_list.append(probs_b.cpu().numpy())

    preds_all, probs_all = np.vstack(preds_list), np.vstack(probs_list)

    mu, sd = scalers[eval_name]['c']['mu'], scalers[eval_name]['c']['sd']
    preds_all_orig = preds_all * sd + mu

    eval_close_orig = all_features[eval_name]['c'].values
    valid_sample_mask = (idx_array + 1 < len(eval_close_orig))
    valid_indices = np.where(valid_sample_mask)[0]

    M = min(200, len(valid_indices))
    chosen = valid_indices[-M:]
    times = [index[idx_array[i]] for i in chosen]
    actuals = np.array([eval_close_orig[idx_array[i]:idx_array[i]+2] for i in chosen])
    preds_plot, probs_plot = preds_all_orig[chosen], probs_all[chosen]

    for h in range(2):
        print(f"Horizon {h+1} metrics: MAE={mae_np(actuals[:, h], preds_plot[:, h]):.6f}, RMSE={rmse_np(actuals[:, h], preds_plot[:, h]):.6f}, MAPE={mape_np(actuals[:, h], preds_plot[:, h]):.3f}%")

    plt.figure(figsize=(12,6))
    plt.plot(times, actuals[:, 0], label='Actual (next close)')
    plt.plot(times, preds_plot[:, 0], label='Predicted (next close)')
    plt.title(f"Predicted vs Actual for {eval_name} (horizon=1)")
    plt.xlabel("Time")
    plt.ylabel("Price (original scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pred_vs_actual_{eval_name}.png')
    plt.show()

def combined_loss_simplified(model, xb, yb):
    preds, _ = model(xb)
    return F.mse_loss(preds, yb)

def main():
    train_norm, eval_norm, eval_name, scalers, all_features, feature_names = load_and_process_data()
    data_arr, index, var_names = build_multi_currency_array(train_norm, feature_names)

    best_seq = find_best_sequence_length(data_arr)

    X_all, idx_array = create_samples(data_arr, seq_len=best_seq, horizon=2)
    y_all = create_targets_for_currency({**{k: all_features[k] for k in train_norm.keys()}, eval_name: all_features[eval_name]}, eval_name, idx_array, horizon=2)
    mu, sd = scalers[eval_name]['c']['mu'], scalers[eval_name]['c']['sd']
    y_all_norm = (y_all - mu) / sd

    eval_close_series = all_features[eval_name]['c'].values
    y_all_cls = np.vstack([(y_all[k] > eval_close_series[idx_array[k]-1]).astype(float) for k in range(len(idx_array))])

    X_train, X_valid, y_train_reg, y_valid_reg, y_train_cls, y_valid_cls = train_test_split(X_all, y_all_norm, y_all_cls, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

    best_model_state = train_model(X_train, y_train_reg, y_train_cls, X_valid, y_valid_reg, y_valid_cls, data_arr.shape[0], best_seq)

    torch.save({'model_state_dict': best_model_state, 'scalers': scalers, 'feature_names': feature_names, 'seq_len': best_seq, 'var_names': var_names}, MODEL_SAVE_PATH)
    print(f"Saved trained model to {MODEL_SAVE_PATH}")

    evaluate_and_plot(best_model_state, scalers, feature_names, best_seq, var_names, X_all, idx_array, eval_name, all_features, index)

if __name__ == '__main__':
    main()