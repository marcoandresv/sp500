import os
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web

# Output directory
DATA_DIR = "data/economic_indicators"
os.makedirs(DATA_DIR, exist_ok=True)

# Define economic indicators from FRED
FRED_SERIES = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "INDPRO": "Industrial Production",
    "FEDFUNDS": "Federal Funds Rate",
    "PCE": "Personal Consumption Expenditures",
    "GS10": "10-Year Treasury Constant Maturity Rate",
}

start_date = datetime(2015, 1, 1)
end_date = datetime.today()

# Download and save each indicator
dfs = []
for code, name in FRED_SERIES.items():
    print(f"Retrieving {name} ({code})...")
    df = web.DataReader(code, "fred", start_date, end_date)
    df.rename(columns={code: name}, inplace=True)
    dfs.append(df)
    df.to_csv(f"{DATA_DIR}/{code}.csv")

# Merge all into a single DataFrame
economic_df = pd.concat(dfs, axis=1)
economic_df.dropna(inplace=True)  # Drop rows with missing data
economic_df.to_csv("data/economic_indicators.csv")

print("✅ All data retrieved and saved.")
