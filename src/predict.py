import torch
import numpy as np
from src.model import ParallelCNNLSTM
from src.config import DEVICE, HORIZON

def predict_future(model_state, scalers, feature_names, seq_len, var_names, last_sequence, horizon=HORIZON):
    """
    Predicts the next N steps into the future.

    Args:
        model_state (dict): The state dictionary of the trained model.
        scalers (dict): The scalers used for normalization.
        feature_names (list): The list of feature names.
        seq_len (int): The sequence length used for training.
        var_names (list): The list of variable names.
        last_sequence (np.ndarray): The most recent sequence of data.
        horizon (int): The number of steps to predict into the future.

    Returns:
        np.ndarray: The predicted future values.
    """
    n_vars = last_sequence.shape[0]
    model = ParallelCNNLSTM(n_vars=n_vars, seq_len=seq_len, hidden_dim=128, horizon=horizon)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    X_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds, _ = model(X_tensor)

    preds_np = preds.cpu().numpy()

    # Inverse transform the predictions
    mu = scalers[var_names[0].split('.')[0]]['c']['mu']
    sd = scalers[var_names[0].split('.')[0]]['c']['sd']
    preds_orig = preds_np * sd + mu

    return preds_orig.flatten()