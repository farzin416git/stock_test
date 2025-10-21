import torch
import numpy as np
from src.config import MODEL_SAVE_PATH, RANDOM_SEED
from src.data_processing import load_and_process_data, build_multi_currency_array, create_samples, create_targets_for_currency
from src.train import find_best_sequence_length, train_model
from src.evaluate import evaluate_and_plot
from sklearn.model_selection import train_test_split

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_norm, _, eval_name, scalers, all_features, feature_names = load_and_process_data()
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