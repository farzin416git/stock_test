import warnings
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Suppress warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """Loads data, adds necessary columns, and returns a DataFrame."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

    # Convert relevant columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(np.float32)

    # Convert 'Date' to datetime and create a time index
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data['time_idx'] = (data['Date'] - data['Date'].min()).dt.days

    # Add a group column - necessary for TimeSeriesDataSet
    data['group'] = 'USDT_IRT'

    return data

def main(file_path='USDT_IRT.csv', epochs=10):
    """Main function to run the stock forecasting process."""
    data = load_and_prepare_data(file_path)
    if data is None:
        return

    # Define the forecast horizon
    max_prediction_length = 30
    # Define the lookback window
    max_encoder_length = 90

    # Split data into training and validation sets
    training_cutoff = data['time_idx'].max() - max_prediction_length
    training_data = data[data['time_idx'] <= training_cutoff]
    validation_data = data[data['time_idx'] > training_cutoff - max_encoder_length]

    # Create the TimeSeriesDataSet
    training_dataset = TimeSeriesDataSet(
        training_data,
        time_idx='time_idx',
        target='Close',
        group_ids=['group'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['group'],
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=['Open', 'High', 'Low', 'Close', 'Volume'],
        target_normalizer=GroupNormalizer(groups=['group'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, validation_data, predict=True, stop_randomization=True
    )

    # Create dataloaders
    batch_size = 128
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    # Configure the model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # Quantile-based uncertainty
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Configure the trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
    )

    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Make predictions
    # Forecast for tomorrow
    tomorrow_prediction = best_tft.predict(
        training_dataset.filter(lambda x: x.time_idx_last == training_cutoff)
    )
    tomorrow_close = tomorrow_prediction[0][0].item()

    # Forecast for the next week and month
    forecast_data = training_dataset.filter(lambda x: x.time_idx_last == training_cutoff)

    # Predict for 30 days
    predictions = best_tft.predict(forecast_data, mode="prediction", return_x=True)

    # Extract the 30-day forecast
    forecast_30_days = predictions.output[0]

    # Calculate averages
    avg_week = forecast_30_days[:7].mean().item()
    avg_month = forecast_30_days.mean().item()

    # Print predictions
    print("\n--- Forecasts ---")
    print(f"Tomorrow's Closing Price: {tomorrow_close:.2f}")
    print(f"Next Week's Average Closing Price: {avg_week:.2f}")
    print(f"Next Month's Average Closing Price: {avg_month:.2f}")
    print("-----------------\n")

if __name__ == '__main__':
    # You can configure the number of epochs here
    main(epochs=1) # Using 1 epoch for quick testing. Increase for better results.
