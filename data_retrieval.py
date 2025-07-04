import os
from datetime import datetime, timedelta

import mysql.connector
import pandas as pd
import pandas_datareader.data as web
from sqlalchemy import create_engine

# --- CONFIGURATION ---
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "spdata"
MYSQL_TABLE = "sp500"

# Define output path
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "SP500.csv")

SP500_CODE = "SP500"
SP500_NAME = "S&P 500 Index"
DEFAULT_START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Check if CSV exists
if os.path.exists(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    existing_df["DATE"] = pd.to_datetime(existing_df["DATE"])
    last_date = existing_df["DATE"].max()
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(
        f"🔍 Found existing file. Appending new data from {start_date} to {END_DATE}..."
    )
else:
    existing_df = pd.DataFrame()
    start_date = DEFAULT_START_DATE
    print(
        f"📄 No existing file found. Downloading full data from {DEFAULT_START_DATE} to {END_DATE}..."
    )

# Download new data
new_df = web.DataReader(SP500_CODE, "fred", start_date, END_DATE)
new_df = new_df.reset_index()
new_df.rename(columns={SP500_CODE: SP500_NAME}, inplace=True)

# Fill missing values
missing_before = new_df[SP500_NAME].isna().sum()
new_df[SP500_NAME] = new_df[SP500_NAME].interpolate(
    method="linear", limit_direction="both"
)
new_df[SP500_NAME] = new_df[SP500_NAME].round(2)
missing_after = new_df[SP500_NAME].isna().sum()
print(
    f"🛠️ Filled {missing_before} missing values in new data, now {missing_after} remain."
)

# Combine with existing
if not existing_df.empty:
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset="DATE", keep="last", inplace=True)
else:
    combined_df = new_df

# Save to CSV
combined_df.to_csv(CSV_PATH, index=False)
print(f"✅ Updated and saved S&P 500 data to: {CSV_PATH}")

# --- SAVE TO MYSQL ---
print("💾 Saving to MySQL database...")

# Create the database if it doesn't exist
conn = mysql.connector.connect(
    host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD
)
cursor = conn.cursor()
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
cursor.close()
conn.close()

# Use SQLAlchemy to connect and write data
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# Save to SQL
combined_df.to_sql(
    MYSQL_TABLE, con=engine, index=False, if_exists="replace"
)  # or 'append'
print(f"✅ Data saved to MySQL table `{MYSQL_DB}.{MYSQL_TABLE}`")

# Preview
print(combined_df.tail())
