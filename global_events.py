import os
from datetime import datetime, timedelta

import pandas as pd


def create_event(name, start_date, end_date=None):
    """Return a single-row DataFrame with start and end dates."""
    return pd.DataFrame(
        [
            {
                "event": name,
                "start_date": start_date,
                "end_date": (
                    end_date if end_date else datetime.today().strftime("%Y-%m-%d")
                ),
            }
        ]
    )


def generate_global_events(sp500_path="data/SP500.csv"):
    events = []

    # Get last known date from SP500 data (minus 7 days for safety)
    if os.path.exists(sp500_path):
        sp500 = pd.read_csv(sp500_path, parse_dates=["DATE"])
        cutoff = (sp500["DATE"].max() - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        cutoff = datetime.today().strftime("%Y-%m-%d")

    # Fixed event ranges
    events.append(create_event("covid_pandemic", "2020-03-01", "2021-12-31"))
    events.append(create_event("fed_rate_hike", "2022-01-01", "2023-06-30"))
    events.append(create_event("banking_panic_2023", "2023-03-08", "2023-03-20"))

    # events.append(
    #     create_event("us_tariff_2025", "2025-02-01", "2025-04-04")
    # )  # US tariffs as one event

    events.append(
        create_event("us_tariff_feb_2025", "2025-02-01", "2025-02-04")
    )  # US tariffs as several events
    events.append(create_event("us_tariff_mar_2025", "2025-03-03", "2025-03-03"))
    events.append(create_event("us_tariff_apr_2025", "2025-04-02", "2025-04-04"))

    # Ongoing events capped at cutoff
    events.append(create_event("ukraine_war", "2022-02-24", cutoff))
    events.append(create_event("ai_boom_2023", "2023-05-01", cutoff))

    df = pd.concat(events, ignore_index=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_events = generate_global_events()
    df_events.to_csv("data/global_events.csv", index=False)
    print(
        f"✅ Global events saved to 'data/global_events.csv' with {len(df_events)} rows."
    )
