import json
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, NBeats
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAPE
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(json_path='data.json'):
    """
    Loads data from a JSON file and prepares it for the N-BEATS model.

    Args:
        json_path (str): The path to the input JSON file.

    Returns:
        pandas.DataFrame: A DataFrame with historical data.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame({
        'Date': pd.to_datetime(data['t'], unit='s'),
        'Open': data['o'],
        'High': data['h'],
        'Low': data['l'],
        'Close': data['c'],
        'Volume': data['v']
    })

    # Convert numeric columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)

    df.rename(columns={'Close': 'value'}, inplace=True)

    # Add a time index
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days

    # Add a dummy group column
    df['group'] = 'USDT_IRT'

    return df

if __name__ == '__main__':
    # Set forecast horizon
    forecast_horizon = 30

    # Load and prepare data
    data_df = load_and_prepare_data()

    # NOTE: The N-BEATS model in pytorch-forecasting is a univariate model and does not support covariates.
    # Therefore, we can only use the target 'value' (Close price) as a time-varying feature.
    # The requirement to use OHLCV as features cannot be met with this specific model implementation.
    # For a model that supports covariates, consider using N-HiTS from the same library.

    # Define the training dataset
    max_encoder_length = 30
    training_cutoff = data_df["time_idx"].max() - forecast_horizon

    training_dataset = TimeSeriesDataSet(
        data_df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=forecast_horizon,
        static_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
    )

    # Create validation dataset and dataloaders
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data_df, predict=True, stop_randomization=True)
    batch_size = 128
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Define the N-BEATS model
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        gradient_clip_val=0.1,
        limit_train_batches=30,
        limit_val_batches=3,
        callbacks=[early_stop_callback],
        logger=False,
        enable_checkpointing=False
    )

    nbeats = NBeats.from_dataset(
        training_dataset,
        learning_rate=3e-2,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=0.1,
    )

    # Train the model
    print("Training the N-BEATS model...")
    trainer.fit(
        nbeats,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Evaluate on validation data
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = nbeats.predict(val_dataloader)
    mape = MAPE()(predictions, actuals)
    print(f"MAPE on validation data: {mape.item():.2f}")

    # Forecast for the next 30 days
    # The predict method can take a dataframe and will predict the future of the last sequence
    predictions = nbeats.predict(data_df)
    forecast_values = predictions[-1].numpy()

    # Create date range for the forecast
    last_date = data_df['Date'].max()
    forecast_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, forecast_horizon + 1)])

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data_df['Date'], data_df['value'], label='Historical Prices', color='blue')
    plt.plot(forecast_dates, forecast_values, label='Forecasted Prices', color='orange')
    plt.title('USDT/IRT Price Forecast with N-BEATS')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = 'forecast_plot.png'
    plt.savefig(plot_filename)
    print(f"Forecast plot saved as {plot_filename}")

    # Show the plot
    plt.show()
