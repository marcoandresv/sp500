import os
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web

# Define the output path
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define parameters
SP500_CODE = "SP500"
SP500_NAME = "S&P 500 Index"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

print(f"\n📈 Downloading {SP500_NAME} data from FRED...")

# Download data
df = web.DataReader(SP500_CODE, "fred", START_DATE, END_DATE)
df = df.reset_index()
df.rename(columns={SP500_CODE: SP500_NAME}, inplace=True)

# Fill missing values using average of previous and next values
missing_before = df[SP500_NAME].isna().sum()
df[SP500_NAME] = df[SP500_NAME].interpolate(method="linear", limit_direction="both")

# Round values to 2 decimal places
df[SP500_NAME] = df[SP500_NAME].round(2)
missing_after = df[SP500_NAME].isna().sum()

print(
    f"🛠️ Filled {missing_before} missing values in '{SP500_NAME}', now {missing_after} remain."
)

# Save to CSV
output_path = os.path.join(DATA_DIR, "SP500.csv")
df.to_csv(output_path, index=False)

print(f"✅ Saved cleaned S&P 500 daily data to: {output_path}")
print(df.head())
