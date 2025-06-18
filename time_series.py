import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

# loading data
df = pd.read_csv("data/SP500.csv")
df.head()


df["DATE"] = pd.to_datetime(df["DATE"])
df = df.rename(
    columns={"DATE": "ds", "S&P 500 Index": "y"}
)  # this is because of how prophet expects the columns to be

# plotting raw data
plt.figure(figsize=(12, 5))
plt.plot(df["ds"], df["y"])
plt.title("S&P 500 Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()


# initializing fit prophet model
model = Prophet(
    daily_seasonality=True, yearly_seasonality=True, seasonality_mode="multiplicative"
)
model.fit(df)

# create future dataframe
future = model.make_future_dataframe(periods=180)  # 180 days into the future
forecast = model.predict(future)

# plot forecast
fig1 = model.plot(forecast)
plt.title("S&P 500 Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.show()

# plot forecast components
fig2 = model.plot_components(forecast)
plt.show()


forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_df.columns = ["date", "prediction", "lower_bound", "upper_bound"]

engine = create_engine("mysql+pymysql://root:password@localhost/timeseries")
forecast_df.to_sql("sp500_forecast", con=engine, if_exists="replace", index=False)
