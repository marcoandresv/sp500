import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

# Load data
df = pd.read_csv("data/SP500.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.rename(columns={"DATE": "ds", "S&P 500 Index": "y"})

# Plot raw data
plt.figure(figsize=(12, 5))
plt.plot(df["ds"], df["y"])
plt.title("S&P 500 Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()

# Initialize Prophet with improved parameters
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="additive",  # More stable for financial data
    changepoint_prior_scale=0.01,  # Tighter trend control
    n_changepoints=100,  # More flexibility to adapt to long-term data
)

# Add quarterly seasonality
model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

# Fit model
model.fit(df)

# Forecast into the future
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("S&P 500 Forecast with Prophet (Tightened Trend)")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()

# Plot components (trend, yearly, quarterly, etc.)
fig2 = model.plot_components(forecast)
plt.show()

# Export forecast to MySQL
forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_df.columns = ["date", "prediction", "lower_bound", "upper_bound"]

engine = create_engine("mysql+pymysql://root:password@localhost/timeseries")
forecast_df.to_sql("sp500_forecast", con=engine, if_exists="replace", index=False)
