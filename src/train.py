import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from tsai.all import get_ts_dls, TSStandardize
from fastai.learner import Learner
from fastai.callback.all import SaveModelCallback
from src.model import ParallelCNNLSTM, combined_loss
from src.config import CANDIDATE_SEQ_LENS, BATCH_SIZE, LEARNING_RATE, DEVICE, RANDOM_SEED, EPOCHS
from src.data_processing import create_samples, create_targets_for_currency
import torch.nn.functional as F

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
        train_losses = []
        for xb, yregb, yclsb in train_loader:
            optimizer.zero_grad()
            preds, probs = model(xb)
            loss = combined_loss(preds, probs, yregb, yclsb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = [combined_loss(model(xb)[0], model(xb)[1], yregb, yclsb) for xb, yregb, yclsb in valid_loader]

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_epoch_val:
            best_epoch_val, best_model_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_model_state