import time

import pandas as pd
import requests


DATA_PATH = "data/processed/train.csv"
API_URL = "http://localhost:8000/predict"
SAMPLE_SIZE = 100

FEATURES = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "estimated_lead_days",
    "item_count",
    "total_price",
    "total_freight_value",
    "customer_state",
]


def main():
    df = pd.read_csv(DATA_PATH)

    samples = df[FEATURES].sample(
        n=SAMPLE_SIZE,
        random_state=42,
    )

    successful = 0
    failed = 0

    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        payload = row.to_dict()

        response = requests.post(
            API_URL,
            json=payload,
            timeout=15,
        )

        if response.status_code == 200:
            successful += 1
        else:
            failed += 1
            print(
                f"Request {index} failed: "
                f"{response.status_code} - {response.text}"
            )

        time.sleep(0.05)

    print("\nTest prediction generation completed.")
    print("Successful:", successful)
    print("Failed:", failed)


if __name__ == "__main__":
    main()