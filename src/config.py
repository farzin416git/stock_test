import torch

DATA_CSV_FOLDER = './data_csv'
EVAL_CSV = './data_csv/eval_currency.csv'
MODEL_SAVE_PATH = './trained_model.pth'
CANDIDATE_SEQ_LENS = [8, 16, 24]
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
HORIZON = 2 # Number of future steps to predict