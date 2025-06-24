import os

import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

# ─────────────────────────────────────────────────────────────────────────────
# Load S&P 500 data
df = pd.read_csv("data/SP500.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.rename(columns={"DATE": "ds", "S&P 500 Index": "y"})

# ─────────────────────────────────────────────────────────────────────────────
# Load and expand global events
events_path = "data/global_events.csv"
if os.path.exists(events_path):
    events = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
    expanded = []
    for _, row in events.iterrows():
        for date in pd.date_range(start=row["start_date"], end=row["end_date"]):
            expanded.append({"ds": date, "holiday": row["event"]})
    holidays_df = pd.DataFrame(expanded)
    print(f"✅ Loaded {len(holidays_df)} global event dates.")
else:
    holidays_df = None
    print("⚠️ No global_events.csv found. Proceeding without event effects.")

# ─────────────────────────────────────────────────────────────────────────────
# Plot raw data
"""
plt.figure(figsize=(12, 5))
plt.plot(df["ds"], df["y"])
plt.title("S&P 500 Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Prophet
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="additive",  # used for financial data
    changepoint_prior_scale=0.05,  # tighter trend
    seasonality_prior_scale=5,  # reduce holiday exaggeration
    n_changepoints=100,  # adds flexibility tto adapt to longterm data
    holidays=holidays_df,
)
model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

# Fit model
model.fit(df)

# Forecast 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# ─────────────────────────────────────────────────────────────────────────────
# Plot forecast
fig1 = model.plot(forecast)
plt.title("S&P 500 Forecast with Prophet (Tightened Trend + Global Events)")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
# plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Export forecast
forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_df.columns = ["date", "prediction", "lower_bound", "upper_bound"]

engine = create_engine("mysql+pymysql://root:password@localhost/timeseries")
forecast_df.to_sql("sp500_forecast", con=engine, if_exists="replace", index=False)
