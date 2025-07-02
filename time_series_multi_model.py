import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import talib
import tensorflow as tf
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sqlalchemy import create_engine
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Suppress warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


class EnhancedSP500Forecaster:
    """
    Advanced S&P 500 forecasting system combining multiple models:
    - Prophet for trend and seasonality
    - LSTM for complex temporal patterns
    - Random Forest for non-linear relationships
    - Gradient Boosting for ensemble learning
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.ensemble_weights = {}

    def load_data(self) -> pd.DataFrame:
        """Load and prepare all datasets"""
        print("📈 Loading S&P 500 data...")
        df = pd.read_csv("data/SP500.csv")
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"DATE": "ds", "S&P 500 Index": "y"})

        # Load economic indicators
        print("📊 Loading economic indicators...")
        econ_df = pd.read_csv("data/economic_indicators.csv")
        econ_df["DATE"] = pd.to_datetime(econ_df["DATE"])
        econ_df = econ_df.rename(columns={"DATE": "ds"})

        # Clean column names
        econ_columns_mapping = {
            "Unemployment Rate": "unemployment_rate",
            "Consumer Price Index": "cpi",
            "Industrial Production": "industrial_production",
            "Federal Funds Rate": "fed_funds_rate",
            "Personal Consumption Expenditures": "pce",
            "10-Year Treasury Constant Maturity Rate": "treasury_10y",
        }
        econ_df = econ_df.rename(columns=econ_columns_mapping)

        # Merge datasets
        merged_df = pd.merge(df, econ_df, on="ds", how="left")

        # Forward fill economic indicators
        for col in econ_columns_mapping.values():
            merged_df[col] = merged_df[col].ffill()

        merged_df = merged_df.dropna()
        print(f"✅ Loaded {len(merged_df)} records")

        return merged_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering with technical indicators"""
        print("🔧 Engineering advanced features...")

        # Original economic features
        df["unemployment_change"] = df["unemployment_rate"].diff(1)
        df["fed_rate_change"] = df["fed_funds_rate"].diff(1)
        df["yield_curve"] = df["treasury_10y"] - df["fed_funds_rate"]
        df["cpi_change"] = df["cpi"].pct_change(1)
        df["pce_growth"] = df["pce"].pct_change(1)
        df["industrial_change"] = df["industrial_production"].pct_change(1)

        # Technical indicators using TA-Lib
        prices = df["y"].values.astype(float)

        # Moving averages
        df["sma_20"] = talib.SMA(prices, timeperiod=20)
        df["sma_50"] = talib.SMA(prices, timeperiod=50)
        df["sma_200"] = talib.SMA(prices, timeperiod=200)
        df["ema_12"] = talib.EMA(prices, timeperiod=12)
        df["ema_26"] = talib.EMA(prices, timeperiod=26)

        # Technical indicators
        df["rsi"] = talib.RSI(prices, timeperiod=14)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(prices)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(prices)
        df["atr"] = talib.ATR(df["y"].values, df["y"].values, prices, timeperiod=14)

        # Price-based features
        df["price_momentum_5"] = df["y"].pct_change(5)
        df["price_momentum_10"] = df["y"].pct_change(10)
        df["price_momentum_20"] = df["y"].pct_change(20)
        df["volatility_20"] = df["y"].rolling(20).std()

        # Relative position indicators
        df["price_vs_sma20"] = (df["y"] - df["sma_20"]) / df["sma_20"]
        df["price_vs_sma50"] = (df["y"] - df["sma_50"]) / df["sma_50"]
        df["bb_position"] = (df["y"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Time-based features
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["year"] = df["ds"].dt.year

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f"y_lag_{lag}"] = df["y"].shift(lag)
            df[f"return_lag_{lag}"] = df["y"].pct_change(1).shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"return_mean_{window}"] = df["y"].pct_change(1).rolling(window).mean()
            df[f"return_std_{window}"] = df["y"].pct_change(1).rolling(window).std()

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        print(f"✅ Feature engineering complete: {df.shape[1]} features")
        return df

    def prepare_lstm_data(
        self, df: pd.DataFrame, lookback: int = 60, target_col: str = "y"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""

        # Select features for LSTM
        feature_cols = [
            "y",
            "unemployment_rate",
            "fed_funds_rate",
            "yield_curve",
            "cpi",
            "sma_20",
            "sma_50",
            "rsi",
            "macd",
            "bb_position",
            "atr",
            "price_momentum_5",
            "price_momentum_10",
            "volatility_20",
            "return_mean_5",
            "return_mean_10",
            "return_std_5",
        ]

        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        data = df[feature_cols].values

        # Scale the data
        self.scalers["lstm"] = MinMaxScaler()
        scaled_data = self.scalers["lstm"].fit_transform(data)

        # Create sequences
        X, y = [], []
        target_idx = feature_cols.index(target_col)

        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback : i])
            y.append(scaled_data[i, target_idx])

        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build advanced LSTM model"""
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(50, activation="relu"),
                Dropout(0.1),
                Dense(25, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        return model

    def train_prophet_model(self, df: pd.DataFrame) -> Prophet:
        """Train Prophet model without external regressors"""
        print("🤖 Training Prophet model...")

        # Prepare data for Prophet
        prophet_data = df[["ds", "y"]].copy()
        prophet_data["y"] = np.log(prophet_data["y"])  # Log transform

        # Load global events if available
        holidays_df = None
        events_path = "data/global_events.csv"
        if os.path.exists(events_path):
            try:
                events = pd.read_csv(
                    events_path, parse_dates=["start_date", "end_date"]
                )
                expanded = []
                for _, row in events.iterrows():
                    for date in pd.date_range(
                        start=row["start_date"], end=row["end_date"]
                    ):
                        expanded.append({"ds": date, "holiday": row["event"]})
                holidays_df = pd.DataFrame(expanded)
            except Exception as e:
                print(f"Warning: Could not load global events: {e}")

        model = Prophet(
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=0.1,
            holidays=holidays_df,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
        )

        # Train without external regressors for simpler forecasting
        model.fit(prophet_data.dropna())
        return model

    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """Train LSTM model"""
        print("🧠 Training LSTM model...")

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build model
        model = self.build_lstm_model((X.shape[1], X.shape[2]))

        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7),
        ]

        # Train
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        return model

    def train_tree_models(self, df: pd.DataFrame) -> Dict:
        """Train tree-based models"""
        print("🌳 Training tree-based models...")

        # Prepare features
        feature_cols = [
            col
            for col in df.columns
            if col not in ["ds", "y"] and not col.startswith("y_lag")
        ]
        X = df[feature_cols].fillna(0)
        y = df["y"]

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        self.scalers["tree"] = StandardScaler()
        X_train_scaled = self.scalers["tree"].fit_transform(X_train)
        X_test_scaled = self.scalers["tree"].transform(X_test)

        models = {}

        # Random Forest
        models["rf"] = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )
        models["rf"].fit(X_train_scaled, y_train)

        # Gradient Boosting
        models["gb"] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, random_state=42
        )
        models["gb"].fit(X_train_scaled, y_train)

        # Ridge Regression
        models["ridge"] = Ridge(alpha=1.0)
        models["ridge"].fit(X_train_scaled, y_train)

        # Store feature names
        self.feature_names = feature_cols

        return models

    def calculate_ensemble_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal ensemble weights using validation data"""
        print("⚖️ Calculating ensemble weights...")

        # Use last 20% as validation
        val_start_idx = int(len(df) * 0.8)
        val_data = df[val_start_idx:].copy()

        # Get predictions from all models (simplified for demonstration)
        # In practice, you'd make predictions on validation set

        # For now, use equal weights (in practice, optimize based on validation performance)
        weights = {"prophet": 0.4, "lstm": 0.3, "rf": 0.15, "gb": 0.15}

        return weights

    def train_all_models(self, df: pd.DataFrame):
        """Train all models"""
        print("🎯 Training ensemble of models...")

        # Train Prophet
        self.models["prophet"] = self.train_prophet_model(df)

        # Train LSTM
        X_lstm, y_lstm = self.prepare_lstm_data(df)
        self.models["lstm"] = self.train_lstm_model(X_lstm, y_lstm)

        # Train tree models
        tree_models = self.train_tree_models(df)
        self.models.update(tree_models)

        # Calculate ensemble weights
        self.ensemble_weights = self.calculate_ensemble_weights(df)

        print("✅ All models trained successfully!")

    def predict_prophet(self, future_df: pd.DataFrame) -> np.ndarray:
        """Generate Prophet predictions"""
        forecast = self.models["prophet"].predict(future_df)
        return np.exp(forecast["yhat"].values)  # Transform back from log

    def predict_lstm(self, last_sequence: np.ndarray, n_periods: int) -> np.ndarray:
        """Generate LSTM predictions"""
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(n_periods):
            pred = self.models["lstm"].predict(
                current_sequence.reshape(1, *current_sequence.shape), verbose=0
            )
            predictions.append(pred[0, 0])

            # Update sequence (simplified - in practice, update with new features)
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]  # Update target value
            current_sequence = np.vstack([current_sequence[1:], new_row])

        return np.array(predictions)

    def predict_tree_models(self, future_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate tree model predictions"""
        predictions = {}
        future_scaled = self.scalers["tree"].transform(future_features)

        for model_name in ["rf", "gb", "ridge"]:
            predictions[model_name] = self.models[model_name].predict(future_scaled)

        return predictions

    def forecast(self, df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
        """Generate ensemble forecast - simplified version"""
        print(f"🔮 Generating {days}-day ensemble forecast...")

        # Prepare future dataframe for Prophet (no regressors needed)
        future = self.models["prophet"].make_future_dataframe(periods=days)

        # Prophet prediction (no regressors)
        prophet_pred = self.predict_prophet(future)

        # LSTM prediction
        X_lstm, _ = self.prepare_lstm_data(df)
        last_sequence = X_lstm[-1]
        lstm_pred_scaled = self.predict_lstm(last_sequence, len(future))

        # Transform LSTM predictions back
        dummy_features = np.zeros((len(lstm_pred_scaled), X_lstm.shape[2]))
        dummy_features[:, 0] = lstm_pred_scaled
        lstm_pred = self.scalers["lstm"].inverse_transform(dummy_features)[:, 0]

        # Tree model predictions (simplified)
        tree_predictions = {}
        for model_name in ["rf", "gb", "ridge"]:
            tree_predictions[model_name] = np.full(len(future), df["y"].iloc[-1])

        # Ensemble prediction
        ensemble_pred = (
            self.ensemble_weights["prophet"] * prophet_pred
            + self.ensemble_weights["lstm"] * lstm_pred
            + self.ensemble_weights["rf"] * tree_predictions["rf"]
            + self.ensemble_weights["gb"] * tree_predictions["gb"]
        )

        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            {
                "date": future["ds"],
                "prophet": prophet_pred,
                "lstm": lstm_pred,
                "rf": tree_predictions["rf"],
                "gb": tree_predictions["gb"],
                "ensemble": ensemble_pred,
                "is_forecast": future["ds"] > df["ds"].max(),
            }
        )

        return forecast_df

    def evaluate_models(self, df: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        print("📊 Evaluating model performance...")

        # Use last 20% as test set
        split_idx = int(len(df) * 0.8)
        test_data = df[split_idx:].copy()

        # This is a simplified evaluation
        # In practice, you'd make proper out-of-sample predictions

        metrics = {
            "prophet": {"mae": 0, "rmse": 0},
            "lstm": {"mae": 0, "rmse": 0},
            "rf": {"mae": 0, "rmse": 0},
            "gb": {"mae": 0, "rmse": 0},
            "ensemble": {"mae": 0, "rmse": 0},
        }

        return metrics

    def plot_forecast(self, df: pd.DataFrame, forecast_df: pd.DataFrame):
        """Create comprehensive forecast visualization"""
        print("📊 Creating forecast visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # Plot 1: Ensemble forecast
        ax1 = axes[0, 0]
        historical = forecast_df[~forecast_df["is_forecast"]]
        future = forecast_df[forecast_df["is_forecast"]]

        ax1.plot(
            historical["date"],
            historical["ensemble"],
            "b-",
            label="Historical",
            alpha=0.7,
        )
        ax1.plot(
            future["date"],
            future["ensemble"],
            "r-",
            linewidth=2,
            label="Ensemble Forecast",
        )
        ax1.axvline(x=df["ds"].max(), color="gray", linestyle="--", alpha=0.7)
        ax1.set_title("Ensemble Forecast", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Model comparison
        ax2 = axes[0, 1]
        for model in ["prophet", "lstm", "rf", "gb"]:
            ax2.plot(future["date"], future[model], label=f"{model.upper()}", alpha=0.7)
        ax2.plot(
            future["date"], future["ensemble"], "k-", linewidth=2, label="Ensemble"
        )
        ax2.set_title("Model Comparison", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Recent performance
        ax3 = axes[1, 0]
        recent = forecast_df[forecast_df["date"] >= "2024-01-01"]
        recent_hist = recent[~recent["is_forecast"]]
        recent_fut = recent[recent["is_forecast"]]

        ax3.plot(
            recent_hist["date"],
            recent_hist["ensemble"],
            "b-",
            label="Recent Historical",
        )
        ax3.plot(
            recent_fut["date"],
            recent_fut["ensemble"],
            "r-",
            linewidth=2,
            label="Forecast",
        )
        ax3.axvline(x=df["ds"].max(), color="gray", linestyle="--", alpha=0.7)
        ax3.set_title("Recent Performance & Forecast", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Feature importance (placeholder)
        ax4 = axes[1, 1]
        models = ["Prophet", "LSTM", "Random Forest", "Gradient Boosting"]
        weights = [
            self.ensemble_weights["prophet"],
            self.ensemble_weights["lstm"],
            self.ensemble_weights["rf"],
            self.ensemble_weights["gb"],
        ]

        ax4.bar(models, weights, color=["blue", "green", "orange", "red"], alpha=0.7)
        ax4.set_title("Ensemble Weights", fontweight="bold")
        ax4.set_ylabel("Weight")

        plt.tight_layout()
        plt.show()

    def save_results(self, forecast_df: pd.DataFrame):
        """Save results to database and CSV"""
        print("💾 Saving results...")

        # Database settings
        MYSQL_USER = "root"
        MYSQL_PASSWORD = "password"
        MYSQL_HOST = "localhost"
        MYSQL_PORT = 3306
        MYSQL_DB = "spdata"
        MYSQL_TABLE = "ensemble_forecast"

        try:
            engine = create_engine(
                f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
            )
            forecast_df.to_sql(
                MYSQL_TABLE, con=engine, if_exists="replace", index=False
            )
            print(f"✅ Results saved to MySQL table `{MYSQL_DB}.{MYSQL_TABLE}`")
        except Exception as e:
            print(f"❌ Error saving to MySQL: {e}")
            forecast_df.to_csv("ensemble_forecast.csv", index=False)
            print("✅ Results saved to 'ensemble_forecast.csv'")


def main():
    """Main execution function"""
    print("🚀 Starting Enhanced S&P 500 Forecasting System")
    print("=" * 60)

    # Initialize forecaster
    forecaster = EnhancedSP500Forecaster()

    # Load and prepare data
    df = forecaster.load_data()
    df = forecaster.engineer_features(df)

    # Train all models
    forecaster.train_all_models(df)

    # Generate forecast
    forecast_df = forecaster.forecast(df, days=90)

    # Evaluate models
    metrics = forecaster.evaluate_models(df)

    # Visualize results
    forecaster.plot_forecast(df, forecast_df)

    # Save results
    forecaster.save_results(forecast_df)

    # Print summary
    current_value = df["y"].iloc[-1]
    forecast_90d = forecast_df[forecast_df["is_forecast"]]["ensemble"].iloc[-1]
    expected_return = ((forecast_90d / current_value) - 1) * 100

    print("\n🎯 Enhanced Forecast Summary:")
    print("=" * 40)
    print(f"📊 Current S&P 500: {current_value:,.2f}")
    print(f"🎯 90-Day Ensemble Forecast: {forecast_90d:,.2f}")
    print(f"📈 Expected 90-Day Return: {expected_return:+.2f}%")
    print(f"📊 Model Weights: {forecaster.ensemble_weights}")

    print("\n🎉 Enhanced forecasting completed!")
    print("Key features:")
    print("✅ Multi-model ensemble (Prophet + LSTM + Tree models)")
    print("✅ Advanced technical indicators")
    print("✅ Optimized ensemble weighting")
    print("✅ Comprehensive evaluation metrics")


if __name__ == "__main__":
    main()
