import pandas as pd
from sqlalchemy import create_engine

# ─────────────────────────────────────────────────────────────────────────────
# MySQL connection config
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "spdata"
OUTPUT_TABLE = "sp500_combined_data"

# ─────────────────────────────────────────────────────────────────────────────
# Create connection
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load data from all 3 tables
sp500_df = pd.read_sql("SELECT * FROM sp500", con=engine)
forecast_df = pd.read_sql("SELECT * FROM timeseries_forecast", con=engine)
enhanced_df = pd.read_sql("SELECT * FROM enhanced_forecast", con=engine)

# ─────────────────────────────────────────────────────────────────────────────
# Normalize column names to lowercase
sp500_df.columns = [col.lower() for col in sp500_df.columns]
forecast_df.columns = [col.lower() for col in forecast_df.columns]
enhanced_df.columns = [col.lower() for col in enhanced_df.columns]

# Ensure the key column is named 'date' in all DataFrames
for df in [sp500_df, forecast_df, enhanced_df]:
    if "ds" in df.columns:
        df.rename(columns={"ds": "date"}, inplace=True)
    elif "DATE" in df.columns:
        df.rename(columns={"DATE": "date"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# Rename prediction columns to avoid conflicts
forecast_df = forecast_df.rename(
    columns={
        "prediction": "forecast_prediction",
        "lower_bound": "forecast_lower_bound",
        "upper_bound": "forecast_upper_bound",
    }
)

enhanced_df = enhanced_df.rename(
    columns={
        "prediction": "enhanced_prediction",
        "lower_bound": "enhanced_lower_bound",
        "upper_bound": "enhanced_upper_bound",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Outer join all three on 'date'
df_merged = pd.merge(sp500_df, forecast_df, on="date", how="outer")
df_merged = pd.merge(df_merged, enhanced_df, on="date", how="outer")

# ─────────────────────────────────────────────────────────────────────────────
# Save combined table to MySQL
df_merged.to_sql(OUTPUT_TABLE, con=engine, index=False, if_exists="replace")
print(f"✅ Combined table saved as `{OUTPUT_TABLE}` in database `{MYSQL_DB}`")
