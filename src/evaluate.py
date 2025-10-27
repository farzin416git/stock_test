import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import ParallelCNNLSTM
from src.config import BATCH_SIZE, DEVICE, HORIZON

def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape_np(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

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