import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")


def load_and_prepare_data(
    sp500_path="data/SP500.csv", events_path="data/global_events.csv"
):
    """Load S&P 500 data and global events"""

    # Load S&P 500 data
    df = pd.read_csv(sp500_path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={"DATE": "ds", "S&P 500 Index": "y"})

    # Load global events for Prophet
    holidays_df = None
    if os.path.exists(events_path):
        events = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
        expanded = []
        for _, row in events.iterrows():
            for date in pd.date_range(start=row["start_date"], end=row["end_date"]):
                expanded.append({"ds": date, "holiday": row["event"]})
        holidays_df = pd.DataFrame(expanded)
        print(f"✅ Loaded {len(holidays_df)} global event dates.")
    else:
        print("⚠️ No global_events.csv found. Proceeding without event effects.")

    return df, holidays_df


def train_prophet_model(df, holidays_df):
    """Train Prophet model exactly like your original"""

    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5,
        n_changepoints=100,
        holidays=holidays_df,
    )
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

    print("Training Prophet model...")
    model.fit(df)

    # Get Prophet predictions for historical data
    historical_forecast = model.predict(df)

    return model, historical_forecast


def create_prophet_residuals_data(df, historical_forecast):
    """Create data for LSTM to learn Prophet residuals"""

    # Calculate Prophet residuals (actual - predicted)
    residuals_df = df.copy()
    residuals_df["prophet_pred"] = historical_forecast["yhat"].values
    residuals_df["residuals"] = residuals_df["y"] - residuals_df["prophet_pred"]

    # Add simple technical features
    residuals_df["returns"] = residuals_df["y"].pct_change()
    residuals_df["ma_5"] = residuals_df["y"].rolling(window=5).mean()
    residuals_df["ma_20"] = residuals_df["y"].rolling(window=20).mean()
    residuals_df["volatility"] = residuals_df["returns"].rolling(window=20).std()

    # Prophet components as features
    residuals_df["trend"] = historical_forecast["trend"].values
    residuals_df["yearly"] = historical_forecast["yearly"].values
    residuals_df["daily"] = historical_forecast["daily"].values
    residuals_df["quarterly"] = historical_forecast["quarterly"].values

    # Drop rows with NaN values
    residuals_df = residuals_df.dropna()

    return residuals_df


def prepare_lstm_data(residuals_df, seq_length=30, test_size=0.15):
    """Prepare data for LSTM to predict residuals"""

    # Features for LSTM (excluding target)
    feature_cols = [
        "returns",
        "ma_5",
        "ma_20",
        "volatility",
        "trend",
        "yearly",
        "daily",
        "quarterly",
    ]
    target_col = "residuals"

    # Split data
    split_idx = int(len(residuals_df) * (1 - test_size))
    train_data = residuals_df.iloc[:split_idx]
    test_data = residuals_df.iloc[split_idx:]

    # Scale features
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_features_scaled = feature_scaler.fit_transform(train_data[feature_cols])
    test_features_scaled = feature_scaler.transform(test_data[feature_cols])

    train_target_scaled = target_scaler.fit_transform(train_data[[target_col]])
    test_target_scaled = target_scaler.transform(test_data[[target_col]])

    # Create sequences
    def create_sequences(features, targets, seq_length):
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i : i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(
        train_features_scaled, train_target_scaled.flatten(), seq_length
    )
    X_test, y_test = create_sequences(
        test_features_scaled, test_target_scaled.flatten(), seq_length
    )

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_cols


def build_simple_lstm_model(input_shape):
    """Build a simple LSTM model for residual prediction"""

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def forecast_hybrid_model(
    prophet_model,
    lstm_model,
    feature_scaler,
    target_scaler,
    last_sequence,
    feature_cols,
    df,
    holidays_df,
    steps=90,
):
    """Generate hybrid forecasts: Prophet + LSTM residuals with continuity"""

    # Get the last known actual value for continuity
    last_actual_value = df["y"].iloc[-1]
    last_date = df["ds"].iloc[-1]

    print(f"Last known value: {last_actual_value:.2f} on {last_date}")

    # Create future dataframe starting from the day after last known data
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )
    future_df = pd.DataFrame({"ds": future_dates})

    # Get Prophet forecast for future dates only
    # First, we need to create a complete future dataframe that includes historical data
    complete_future = prophet_model.make_future_dataframe(periods=steps)
    prophet_forecast = prophet_model.predict(complete_future)

    # Extract only the future predictions (last 'steps' rows)
    future_prophet = prophet_forecast.tail(steps).reset_index(drop=True)

    # Get the Prophet prediction for the last historical date to calculate adjustment
    historical_complete = prophet_model.predict(df)
    last_prophet_pred = historical_complete["yhat"].iloc[-1]

    # Calculate the adjustment needed to ensure continuity
    continuity_adjustment = last_actual_value - last_prophet_pred
    print(f"Continuity adjustment: {continuity_adjustment:.2f}")

    # Prepare for LSTM predictions
    predictions = []
    current_seq = last_sequence.copy()

    # Get last known features for feature evolution
    last_known_data = df.tail(30)  # Use last 30 days for feature calculation
    last_returns = last_known_data["y"].pct_change().iloc[-1]
    last_ma5 = last_known_data["y"].rolling(5).mean().iloc[-1]
    last_ma20 = last_known_data["y"].rolling(20).mean().iloc[-1]
    last_volatility = last_known_data["y"].pct_change().rolling(20).std().iloc[-1]

    # Store the previous prediction for feature calculation
    prev_prediction = last_actual_value

    for i in range(steps):
        # Predict residual using LSTM
        residual_pred = lstm_model.predict(
            current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]),
            verbose=0,
        )[0][0]
        residual_pred = target_scaler.inverse_transform([[residual_pred]])[0][0]

        # Get Prophet prediction for this day and apply continuity adjustment
        prophet_pred = future_prophet.iloc[i]["yhat"] + continuity_adjustment

        # Combine Prophet prediction with LSTM residual (with dampening)
        # Reduce the residual effect over time to prevent drift
        residual_weight = max(0.3 * (0.95**i), 0.05)  # Exponential decay
        final_prediction = prophet_pred + residual_pred * residual_weight

        predictions.append(final_prediction)

        # Update features for next prediction based on the current prediction
        if i == 0:
            # First prediction: calculate return from last actual value
            current_return = (final_prediction - prev_prediction) / prev_prediction
        else:
            # Subsequent predictions: calculate return from previous prediction
            current_return = (final_prediction - prev_prediction) / prev_prediction

        # Update moving averages (simplified - in practice you'd maintain full history)
        current_ma5 = (last_ma5 * 4 + final_prediction) / 5  # Approximate MA5 update
        current_ma20 = (
            last_ma20 * 19 + final_prediction
        ) / 20  # Approximate MA20 update

        # Update volatility (simplified)
        current_volatility = last_volatility * 0.99 + abs(current_return) * 0.01

        # Get Prophet components for this future date
        trend_val = future_prophet.iloc[i]["trend"]
        yearly_val = (
            future_prophet.iloc[i]["yearly"]
            if "yearly" in future_prophet.columns
            else 0
        )
        daily_val = (
            future_prophet.iloc[i]["daily"] if "daily" in future_prophet.columns else 0
        )
        quarterly_val = (
            future_prophet.iloc[i]["quarterly"]
            if "quarterly" in future_prophet.columns
            else 0
        )

        # Create feature vector for next prediction
        next_features = np.array(
            [
                current_return,
                current_ma5,
                current_ma20,
                current_volatility,
                trend_val,
                yearly_val,
                daily_val,
                quarterly_val,
            ]
        )

        # Scale the features
        next_features_scaled = feature_scaler.transform(next_features.reshape(1, -1))[0]

        # Update sequence for next prediction
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = next_features_scaled

        # Update for next iteration
        prev_prediction = final_prediction
        last_ma5 = current_ma5
        last_ma20 = current_ma20
        last_volatility = current_volatility

    return predictions, future_dates.tolist()


def main():
    """Main execution function"""

    print("Loading data...")
    df, holidays_df = load_and_prepare_data()

    print("Training Prophet model...")
    prophet_model, historical_forecast = train_prophet_model(df, holidays_df)

    print("Preparing residuals data...")
    residuals_df = create_prophet_residuals_data(df, historical_forecast)

    print("Preparing LSTM data...")
    seq_length = 30
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_cols = (
        prepare_lstm_data(residuals_df, seq_length)
    )

    print(f"LSTM training data shape: X={X_train.shape}, y={y_train.shape}")

    print("Building and training LSTM model...")
    lstm_model = build_simple_lstm_model((seq_length, len(feature_cols)))

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    history = lstm_model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )

    print("Evaluating models...")

    # Evaluate LSTM on residuals
    lstm_pred = lstm_model.predict(X_test)
    lstm_pred_rescaled = target_scaler.inverse_transform(lstm_pred)
    y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_pred_rescaled))
    print(f"LSTM Residual RMSE: {lstm_rmse:.2f}")

    # Generate Prophet-only forecast for comparison (with continuity adjustment)
    last_actual_value = df["y"].iloc[-1]
    complete_future = prophet_model.make_future_dataframe(periods=90)
    prophet_forecast_complete = prophet_model.predict(complete_future)

    # Calculate continuity adjustment for Prophet-only forecast
    last_prophet_pred = prophet_forecast_complete.iloc[len(df) - 1]["yhat"]
    prophet_adjustment = last_actual_value - last_prophet_pred

    # Extract future Prophet predictions and apply adjustment
    prophet_future = prophet_forecast_complete.tail(90).copy()
    prophet_future["yhat"] += prophet_adjustment
    prophet_future["yhat_lower"] += prophet_adjustment
    prophet_future["yhat_upper"] += prophet_adjustment

    # Generate hybrid forecast
    last_sequence = X_test[-1] if len(X_test) > 0 else X_train[-1]
    hybrid_predictions, future_dates = forecast_hybrid_model(
        prophet_model,
        lstm_model,
        feature_scaler,
        target_scaler,
        last_sequence,
        feature_cols,
        df,
        holidays_df,
        steps=90,
    )

    # Create results DataFrames
    prophet_results = pd.DataFrame(
        {
            "Date": prophet_future["ds"],
            "Prophet_Forecast": prophet_future["yhat"],
            "Prophet_Lower": prophet_future["yhat_lower"],
            "Prophet_Upper": prophet_future["yhat_upper"],
        }
    )

    hybrid_results = pd.DataFrame(
        {"Date": future_dates, "Hybrid_Forecast": hybrid_predictions}
    )

    # Plotting
    plt.figure(figsize=(20, 12))

    # Plot 1: Prophet vs Hybrid Comparison with Continuity Check
    plt.subplot(2, 3, 1)

    # Plot more historical data to see the connection
    historical_window = 100
    plt.plot(
        df["ds"].tail(historical_window),
        df["y"].tail(historical_window),
        label="Historical",
        alpha=0.8,
        color="blue",
        linewidth=1.5,
    )

    plt.plot(
        prophet_results["Date"],
        prophet_results["Prophet_Forecast"],
        label="Prophet Only",
        color="green",
        linewidth=2,
    )
    plt.plot(
        hybrid_results["Date"],
        hybrid_results["Hybrid_Forecast"],
        label="Hybrid (Prophet + LSTM)",
        color="red",
        linewidth=2,
        linestyle="--",
    )

    # Add a marker at the last historical point
    plt.scatter(
        df["ds"].iloc[-1],
        df["y"].iloc[-1],
        color="black",
        s=100,
        zorder=5,
        label=f'Last Known: {df["y"].iloc[-1]:.0f}',
    )

    # Add markers at first forecast points
    plt.scatter(
        prophet_results["Date"].iloc[0],
        prophet_results["Prophet_Forecast"].iloc[0],
        color="green",
        s=80,
        zorder=5,
    )
    plt.scatter(
        hybrid_results["Date"].iloc[0],
        hybrid_results["Hybrid_Forecast"].iloc[0],
        color="red",
        s=80,
        zorder=5,
    )

    plt.fill_between(
        prophet_results["Date"],
        prophet_results["Prophet_Lower"],
        prophet_results["Prophet_Upper"],
        alpha=0.2,
        color="green",
    )
    plt.title("Prophet vs Hybrid Model Comparison (with Continuity)")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Prophet Components
    plt.subplot(2, 3, 2)
    components = prophet_model.predict(df)
    plt.plot(df["ds"], components["trend"], label="Trend")
    plt.plot(df["ds"], components["yearly"], label="Yearly")
    plt.plot(df["ds"], components["quarterly"], label="Quarterly")
    plt.title("Prophet Components")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Residuals Analysis
    plt.subplot(2, 3, 3)
    plt.plot(residuals_df["ds"], residuals_df["residuals"], alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Prophet Residuals (LSTM Target)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)

    # Plot 4: LSTM Training History
    plt.subplot(2, 3, 4)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Plot 5: Forecast Detail - Focus on Continuity
    plt.subplot(2, 3, 5)
    # Show last 30 days of historical + first 30 days of forecast
    historical_tail = df.tail(30)
    plt.plot(
        historical_tail["ds"],
        historical_tail["y"],
        label="Historical",
        alpha=0.8,
        color="blue",
        linewidth=2,
    )
    plt.plot(
        prophet_results["Date"].head(30),
        prophet_results["Prophet_Forecast"].head(30),
        label="Prophet",
        color="green",
        linewidth=2,
    )
    plt.plot(
        hybrid_results["Date"].head(30),
        hybrid_results["Hybrid_Forecast"].head(30),
        label="Hybrid",
        color="red",
        linewidth=2,
        linestyle="--",
    )

    # Mark the transition point
    plt.axvline(
        x=df["ds"].iloc[-1],
        color="black",
        linestyle=":",
        alpha=0.7,
        label="Last Known Data",
    )
    plt.scatter(df["ds"].iloc[-1], df["y"].iloc[-1], color="black", s=100, zorder=5)

    plt.title("Continuity Check: Last 30 Historical + First 30 Forecast Days")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Index")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Model Statistics with Continuity Info
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, f"Continuity Analysis:", fontsize=12, weight="bold")
    plt.text(0.1, 0.85, f"Last known value: {df['y'].iloc[-1]:.2f}", fontsize=10)
    plt.text(
        0.1,
        0.8,
        f"First Prophet forecast: {prophet_results['Prophet_Forecast'].iloc[0]:.2f}",
        fontsize=10,
    )
    plt.text(
        0.1,
        0.75,
        f"First Hybrid forecast: {hybrid_results['Hybrid_Forecast'].iloc[0]:.2f}",
        fontsize=10,
    )

    prophet_gap = abs(prophet_results["Prophet_Forecast"].iloc[0] - df["y"].iloc[-1])
    hybrid_gap = abs(hybrid_results["Hybrid_Forecast"].iloc[0] - df["y"].iloc[-1])
    plt.text(0.1, 0.7, f"Prophet gap: {prophet_gap:.2f}", fontsize=10)
    plt.text(0.1, 0.65, f"Hybrid gap: {hybrid_gap:.2f}", fontsize=10)

    plt.text(0.1, 0.55, f"Prophet Model Statistics:", fontsize=12, weight="bold")
    plt.text(0.1, 0.5, f"- Uses global events as holidays", fontsize=10)
    plt.text(0.1, 0.45, f"- Quarterly seasonality", fontsize=10)
    plt.text(0.1, 0.4, f"- Changepoint prior: 0.05", fontsize=10)
    plt.text(0.1, 0.35, f"- Seasonality prior: 5", fontsize=10)

    plt.text(0.1, 0.25, f"LSTM Model Statistics:", fontsize=12, weight="bold")
    plt.text(0.1, 0.2, f"- Residual RMSE: {lstm_rmse:.2f}", fontsize=10)
    plt.text(0.1, 0.15, f"- Sequence length: {seq_length}", fontsize=10)
    plt.text(0.1, 0.1, f"- Residual weight decay: 0.3 → 0.05", fontsize=10)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.title("Model Configuration & Continuity")

    plt.tight_layout()
    plt.show()

    # Save results
    os.makedirs("data", exist_ok=True)
    prophet_results.to_csv("data/prophet_forecast.csv", index=False)
    hybrid_results.to_csv("data/hybrid_forecast.csv", index=False)

    # Combined results
    combined_results = prophet_results.merge(hybrid_results, on="Date", how="inner")
    combined_results.to_csv("data/combined_forecasts_improved.csv", index=False)

    # Save to MySQL
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "password"
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_DB = "spdata"
    MYSQL_TABLE = "enhanced_forecast"

    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

    combined_results.to_sql(MYSQL_TABLE, con=engine, if_exists="replace", index=False)
    print(f"✅ Forecast saved to MySQL table `{MYSQL_DB}.{MYSQL_TABLE}`")

    print("\nForecasts saved to:")
    print("- data/prophet_forecast.csv")
    print("- data/hybrid_forecast.csv")
    print("- data/combined_forecasts_improved.csv")

    print(f"\nForecast Summary:")
    print(
        f"Prophet 90-day trend: {prophet_results['Prophet_Forecast'].iloc[-1] - prophet_results['Prophet_Forecast'].iloc[0]:.0f} points"
    )
    print(
        f"Hybrid 90-day trend: {hybrid_results['Hybrid_Forecast'].iloc[-1] - hybrid_results['Hybrid_Forecast'].iloc[0]:.0f} points"
    )

    print(f"\nContinuity Check:")
    print(f"Last known value: {df['y'].iloc[-1]:.2f}")
    print(
        f"First Prophet forecast: {prophet_results['Prophet_Forecast'].iloc[0]:.2f} (gap: {abs(prophet_results['Prophet_Forecast'].iloc[0] - df['y'].iloc[-1]):.2f})"
    )
    print(
        f"First Hybrid forecast: {hybrid_results['Hybrid_Forecast'].iloc[0]:.2f} (gap: {abs(hybrid_results['Hybrid_Forecast'].iloc[0] - df['y'].iloc[-1]):.2f})"
    )

    return prophet_model, lstm_model, prophet_results, hybrid_results


if __name__ == "__main__":
    prophet_model, lstm_model, prophet_results, hybrid_results = main()
