
import time
import logging
from datetime import datetime

from src.monitoring.data_drift import generate_drift_report


# Run drift monitoring every 6 hours
INTERVAL_SECONDS = 6 * 60 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def run_monitoring():
    logging.info("Starting scheduled data drift check.")

    try:
        generate_drift_report()
        logging.info("Data drift check completed successfully.")

    except Exception:
        logging.exception("Data drift check failed.")


def main():
    logging.info("Scheduled drift monitoring started.")
    logging.info("Monitoring interval: 6 hours.")

    while True:
        run_monitoring()

        logging.info(
            "Next drift check scheduled in 6 hours."
        )

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
