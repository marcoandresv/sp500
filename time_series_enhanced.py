import os
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings("ignore", message=".*initial.*")
warnings.filterwarnings("ignore", message=".*seasonality.*")

# ─────────────────────────────────────────────────────────────────────────────
# Load S&P 500 data
print("📈 Loading S&P 500 data...")
df = pd.read_csv("data/SP500.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.rename(columns={"DATE": "ds", "S&P 500 Index": "y"})

# Add log transformation for better trend modeling
df["y_log"] = np.log(df["y"])
print(f"✅ Loaded {len(df)} S&P 500 records from {df['ds'].min()} to {df['ds'].max()}")

# Calculate recent trend to inform model
recent_trend = df["y"].pct_change(30).iloc[-1] * 100
print(f"📊 Recent 30-day trend: {recent_trend:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Load economic indicators
print("\n📊 Loading economic indicators...")
econ_df = pd.read_csv("data/economic_indicators.csv")
econ_df["DATE"] = pd.to_datetime(econ_df["DATE"])
econ_df = econ_df.rename(columns={"DATE": "ds"})

# Clean column names for Prophet
econ_columns_mapping = {
    "Unemployment Rate": "unemployment_rate",
    "Consumer Price Index": "cpi",
    "Industrial Production": "industrial_production",
    "Federal Funds Rate": "fed_funds_rate",
    "Personal Consumption Expenditures": "pce",
    "10-Year Treasury Constant Maturity Rate": "treasury_10y",
}
econ_df = econ_df.rename(columns=econ_columns_mapping)
economic_indicators = list(econ_columns_mapping.values())

print(f"✅ Loaded {len(econ_df)} economic indicator records")
print(f"📊 Economic indicators: {economic_indicators}")

# ─────────────────────────────────────────────────────────────────────────────
# Merge S&P 500 with economic indicators
print("\n🔗 Merging datasets...")
merged_df = pd.merge(df, econ_df, on="ds", how="left")

# Intelligent forward fill for economic indicators (monthly data)
for indicator in economic_indicators:
    merged_df[indicator] = merged_df[indicator].ffill()

# Drop any remaining NaN values
initial_length = len(merged_df)
merged_df = merged_df.dropna()
print(
    f"✅ Merged dataset: {len(merged_df)} records (dropped {initial_length - len(merged_df)} NaN records)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load and expand global events
events_path = "data/global_events.csv"
if os.path.exists(events_path):
    print("\n🌍 Loading global events...")
    events = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
    expanded = []
    for _, row in events.iterrows():
        for date in pd.date_range(start=row["start_date"], end=row["end_date"]):
            expanded.append({"ds": date, "holiday": row["event"]})
    holidays_df = pd.DataFrame(expanded)
    print(
        f"✅ Loaded {len(holidays_df)} global event dates covering {len(events)} events"
    )
else:
    holidays_df = None
    print("⚠️ No global_events.csv found. Proceeding without event effects.")

# ─────────────────────────────────────────────────────────────────────────────
# Improved feature engineering
print("\n🔧 Engineering economic indicator features...")

# Focus on most predictive features for stock market
key_features = []

# 1. Unemployment rate change (inversely correlated with market)
merged_df["unemployment_change"] = merged_df["unemployment_rate"].diff(1)
key_features.append("unemployment_change")

# 2. Fed funds rate change (major market driver)
merged_df["fed_rate_change"] = merged_df["fed_funds_rate"].diff(1)
key_features.append("fed_rate_change")

# 3. Yield curve (10Y - Fed funds rate)
merged_df["yield_curve"] = merged_df["treasury_10y"] - merged_df["fed_funds_rate"]
key_features.append("yield_curve")

# 4. CPI change (inflation pressure)
merged_df["cpi_change"] = merged_df["cpi"].pct_change(1)
key_features.append("cpi_change")

# 5. Economic momentum (PCE growth)
merged_df["pce_growth"] = merged_df["pce"].pct_change(1)
key_features.append("pce_growth")

# 6. Industrial production change
merged_df["industrial_change"] = merged_df["industrial_production"].pct_change(1)
key_features.append("industrial_change")

# Remove outliers and handle infinities
for feature in key_features:
    # Replace infinities with NaN
    merged_df[feature] = merged_df[feature].replace([np.inf, -np.inf], np.nan)
    # Fill NaN with median
    merged_df[feature] = merged_df[feature].fillna(merged_df[feature].median())

    # Cap extreme outliers (beyond 3 standard deviations)
    std_val = merged_df[feature].std()
    mean_val = merged_df[feature].mean()
    merged_df[feature] = merged_df[feature].clip(
        lower=mean_val - 3 * std_val, upper=mean_val + 3 * std_val
    )

# Final cleanup
merged_df = merged_df.dropna()
print(f"✅ Feature engineering complete. Final dataset: {len(merged_df)} records")
print(f"📋 Key economic features: {key_features}")

# ─────────────────────────────────────────────────────────────────────────────
# Initialize improved Prophet model
print("\n🤖 Initializing optimized Prophet model...")

# Use log scale for better trend modeling
model_data = merged_df.copy()
model_data["y"] = model_data["y_log"]  # Use log-transformed values

model = Prophet(
    # Trend parameters - more flexible for long-term growth
    changepoint_prior_scale=0.1,  # Increased flexibility for trend changes
    n_changepoints=50,  # Fewer changepoints for smoother trend
    # Seasonality parameters
    yearly_seasonality=True,
    daily_seasonality=False,  # Remove daily noise
    weekly_seasonality=True,
    seasonality_mode="additive",
    seasonality_prior_scale=0.1,  # Reduce seasonality impact
    # Uncertainty parameters
    interval_width=0.80,
    mcmc_samples=0,  # Faster training
    # Growth parameters
    growth="linear",  # Linear growth assumption
    # Holidays
    holidays=holidays_df,
    holidays_prior_scale=0.1,  # Reduce holiday impact
)

# Add optimized seasonality
model.add_seasonality(name="quarterly", period=91.25, fourier_order=6)
model.add_seasonality(name="monthly", period=30.5, fourier_order=4)

# Add only the most impactful economic regressors
important_regressors = [
    "yield_curve",  # Most important for markets
    "fed_rate_change",  # Fed policy impacts
    "unemployment_change",  # Economic health
    "cpi_change",  # Inflation concerns
]

for regressor in important_regressors:
    # Add with moderate prior scale to avoid overfitting
    model.add_regressor(regressor, prior_scale=0.1, standardize=True)

print(f"✅ Added {len(important_regressors)} key economic regressors")

# ─────────────────────────────────────────────────────────────────────────────
# Fit the model
print("\n🏋️ Training optimized Prophet model...")
model.fit(model_data)
print("✅ Model training completed!")

# ─────────────────────────────────────────────────────────────────────────────
# Create future dataframe with realistic economic assumptions
print("\n🔮 Preparing 90-day forecast with realistic assumptions...")

# Create future dates
future = model.make_future_dataframe(periods=90)

# Merge with historical economic data
future = future.merge(model_data[["ds"] + important_regressors], on="ds", how="left")

# For future economic indicators, use more realistic assumptions
last_date_with_econ = model_data["ds"].iloc[-1]
print(f"📅 Last date with economic data: {last_date_with_econ}")

# Realistic economic assumptions for next 90 days
for regressor in important_regressors:
    last_value = model_data[regressor].iloc[-1]
    recent_trend = model_data[regressor].iloc[-30:].mean()

    if regressor == "yield_curve":
        # Assume yield curve stays relatively stable
        future[regressor] = future[regressor].fillna(last_value)
    elif regressor == "fed_rate_change":
        # Assume no dramatic fed rate changes (typical for 90-day period)
        future[regressor] = future[regressor].fillna(0.0)
    elif regressor == "unemployment_change":
        # Assume gradual improvement continues
        future[regressor] = future[regressor].fillna(min(0.0, recent_trend))
    elif regressor == "cpi_change":
        # Assume moderate inflation continues
        future[regressor] = future[regressor].fillna(max(0.001, recent_trend))

print("✅ Future economic assumptions prepared")

# ─────────────────────────────────────────────────────────────────────────────
# Generate forecast
print("\n🎯 Generating forecast...")
forecast = model.predict(future)

# Transform back from log scale
forecast["yhat"] = np.exp(forecast["yhat"])
forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

print("✅ Forecast generation completed!")

# ─────────────────────────────────────────────────────────────────────────────
# Apply trend adjustment based on historical performance
print("\n📈 Applying trend adjustment...")

# Calculate historical trend (annualized return)
years_of_data = (model_data["ds"].iloc[-1] - model_data["ds"].iloc[0]).days / 365.25
total_return = (np.exp(model_data["y"].iloc[-1]) / np.exp(model_data["y"].iloc[0])) - 1
historical_annual_return = (1 + total_return) ** (1 / years_of_data) - 1

print(f"📊 Historical annual return: {historical_annual_return:.2%}")

# Apply gentle trend adjustment to future predictions
future_mask = forecast["ds"] > model_data["ds"].max()
days_ahead = (forecast.loc[future_mask, "ds"] - model_data["ds"].max()).dt.days

# Conservative trend adjustment (half of historical trend)
trend_adjustment = 1 + (historical_annual_return * 0.5 * days_ahead / 365.25)

forecast.loc[future_mask, "yhat"] *= trend_adjustment
forecast.loc[future_mask, "yhat_lower"] *= trend_adjustment * 0.95
forecast.loc[future_mask, "yhat_upper"] *= trend_adjustment * 1.05

print("✅ Trend adjustment applied")

# ─────────────────────────────────────────────────────────────────────────────
# Create enhanced visualizations
print("\n📊 Creating enhanced visualizations...")

# Main forecast plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Full time series
ax1.plot(
    model_data["ds"],
    np.exp(model_data["y"]),
    "ko",
    markersize=1,
    alpha=0.7,
    label="Historical Data",
)
ax1.plot(forecast["ds"], forecast["yhat"], "b-", linewidth=2, label="Forecast")
ax1.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.3,
    color="blue",
    label="Confidence Interval",
)

# Highlight the forecast period
forecast_start = model_data["ds"].max()
ax1.axvline(
    x=forecast_start, color="red", linestyle="--", alpha=0.7, label="Forecast Start"
)

ax1.set_title(
    "S&P 500 Forecast with Optimized Economic Model", fontsize=16, fontweight="bold"
)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("S&P 500 Index Value", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Zoomed view of recent data and forecast
recent_data = forecast[forecast["ds"] >= "2024-01-01"].copy()
ax2.plot(
    recent_data[recent_data["ds"] <= forecast_start]["ds"],
    recent_data[recent_data["ds"] <= forecast_start]["yhat"],
    "ko",
    markersize=2,
    alpha=0.8,
    label="Recent Historical",
)
ax2.plot(
    recent_data[recent_data["ds"] > forecast_start]["ds"],
    recent_data[recent_data["ds"] > forecast_start]["yhat"],
    "b-",
    linewidth=3,
    label="90-Day Forecast",
)
ax2.fill_between(
    recent_data[recent_data["ds"] > forecast_start]["ds"],
    recent_data[recent_data["ds"] > forecast_start]["yhat_lower"],
    recent_data[recent_data["ds"] > forecast_start]["yhat_upper"],
    alpha=0.3,
    color="blue",
    label="Confidence Interval",
)

ax2.axvline(
    x=forecast_start, color="red", linestyle="--", alpha=0.7, label="Forecast Start"
)
ax2.set_title(
    "Recent Performance and 90-Day Forecast Detail", fontsize=14, fontweight="bold"
)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("S&P 500 Index Value", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced forecast analysis
print("\n🎯 Enhanced Forecast Analysis:")
print("=" * 60)

# Current and forecast values
current_value = np.exp(model_data["y"].iloc[-1])
forecast_30d = (
    forecast[forecast["ds"] == forecast_start + timedelta(days=30)]["yhat"].iloc[0]
    if len(forecast[forecast["ds"] == forecast_start + timedelta(days=30)]) > 0
    else None
)
forecast_60d = (
    forecast[forecast["ds"] == forecast_start + timedelta(days=60)]["yhat"].iloc[0]
    if len(forecast[forecast["ds"] == forecast_start + timedelta(days=60)]) > 0
    else None
)
forecast_90d = forecast[forecast["ds"] > forecast_start]["yhat"].iloc[-1]

print(f"📊 Current S&P 500 Value: {current_value:,.2f}")
print(
    f"🎯 30-Day Forecast: {forecast_30d:,.2f}"
    if forecast_30d
    else "🎯 30-Day Forecast: N/A"
)
print(
    f"🎯 60-Day Forecast: {forecast_60d:,.2f}"
    if forecast_60d
    else "🎯 60-Day Forecast: N/A"
)
print(f"🎯 90-Day Forecast: {forecast_90d:,.2f}")

# Return calculations
if forecast_30d:
    return_30d = ((forecast_30d / current_value) - 1) * 100
    print(f"📈 Expected 30-Day Return: {return_30d:+.2f}%")
if forecast_60d:
    return_60d = ((forecast_60d / current_value) - 1) * 100
    print(f"📈 Expected 60-Day Return: {return_60d:+.2f}%")

return_90d = ((forecast_90d / current_value) - 1) * 100
annualized_forecast = ((forecast_90d / current_value) ** (365.25 / 90)) - 1
print(f"📈 Expected 90-Day Return: {return_90d:+.2f}%")
print(f"📈 Annualized Forecast Return: {annualized_forecast * 100:+.2f}%")

# Confidence intervals
future_forecast = forecast[forecast["ds"] > forecast_start].copy()
confidence_width = (
    (future_forecast["yhat_upper"].iloc[-1] - future_forecast["yhat_lower"].iloc[-1])
    / future_forecast["yhat"].iloc[-1]
    * 100
)

print(
    f"\n🎯 90-Day Confidence Interval: [{future_forecast['yhat_lower'].iloc[-1]:,.2f}, {future_forecast['yhat_upper'].iloc[-1]:,.2f}]"
)
print(f"📊 Confidence Width: ±{confidence_width/2:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Export enhanced forecast
print("\n💾 Preparing enhanced forecast export...")
forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_df.columns = ["date", "prediction", "lower_bound", "upper_bound"]

# Add metadata
forecast_df["is_forecast"] = forecast_df["date"] > model_data["ds"].max()
forecast_df["days_ahead"] = (forecast_df["date"] - model_data["ds"].max()).dt.days
forecast_df["confidence_width"] = (
    forecast_df["upper_bound"] - forecast_df["lower_bound"]
)
forecast_df["expected_return"] = ((forecast_df["prediction"] / current_value) - 1) * 100

# ─────────────────────────────────────────────────────────────────────────────
# Save to MySQL database
print("\n🗄️ Saving enhanced forecast to MySQL database...")

MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "spdata"
MYSQL_TABLE = "timeseries_forecast_optimized"

try:
    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

    forecast_df.to_sql(MYSQL_TABLE, con=engine, if_exists="replace", index=False)
    print(f"✅ Optimized forecast saved to MySQL table `{MYSQL_DB}.{MYSQL_TABLE}`")
    print(f"📊 Total records saved: {len(forecast_df)}")

except Exception as e:
    print(f"❌ Error saving to MySQL: {e}")
    print("💡 Saving to CSV as backup...")
    forecast_df.to_csv("sp500_forecast_optimized.csv", index=False)
    print("✅ Forecast saved to 'sp500_forecast_optimized.csv'")

print("\n🎉 Optimized S&P 500 forecasting completed!")
print("Key improvements:")
print("✅ Log-scale modeling for better trend capture")
print("✅ Focused on most predictive economic indicators")
print("✅ Realistic economic assumptions for forecasting")
print("✅ Trend adjustment based on historical performance")
print("✅ Enhanced confidence interval analysis")
