from datetime import datetime
from pathlib import Path
import sys

from loguru import logger
import pandas as pd
import typer

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(sys.stderr, level="WARNING")


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

feature_directory = Path("/codeexecution/data")
prediction_path = Path("/codeexecution/prediction.csv")


def main(prediction_time: datetime):
    logger.info("Processing {}", prediction_time)
    logger.debug("Copy partial submission format to prediction.")
    submission_format = pd.read_csv(
        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    )

    # Read features, process them, run your model
    prediction = submission_format.copy()

    prediction.to_csv(prediction_path, date_format=DATETIME_FORMAT)


if __name__ == "__main__":
    typer.run(main)
