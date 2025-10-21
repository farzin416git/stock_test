import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc(x)))
        return x

class LSTMBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(F.relu(self.fc(last)))
        return last

class ParallelCNNLSTM(nn.Module):
    def __init__(self, n_vars, seq_len, hidden_dim=128, horizon=2):
        super().__init__()
        self.cnn = CNNBranch(in_channels=n_vars, seq_len=seq_len, hidden_dim=hidden_dim)
        self.lstm = LSTMBranch(in_channels=n_vars, hidden_dim=hidden_dim)
        self.combined_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.reg_head = nn.Linear(hidden_dim, horizon)
        self.cls_head = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        c = self.cnn(x)
        l = self.lstm(x)
        stacked = torch.cat([c, l], dim=1)
        h = F.relu(self.combined_fc(stacked))
        preds = self.reg_head(h)
        logits = self.cls_head(h)
        probs = torch.sigmoid(logits)
        return preds, probs

def combined_loss(preds: torch.Tensor, probs: torch.Tensor, y_reg: torch.Tensor, y_cls: torch.Tensor, alpha=1.0, beta=0.5):
    mse = F.mse_loss(preds, y_reg)
    bce = F.binary_cross_entropy(probs, y_cls)
    return alpha * mse + beta * bce