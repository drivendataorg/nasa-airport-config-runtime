from datetime import datetime
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:S"

feature_directory = Path("/codeexecution/data")
prediction_directory = Path("/codeexecution/predictions")

lookaheads = pd.timedelta_range("30min", "6H", freq="30min").total_seconds() // 60


def main(prediction_timestamp: datetime):
    logger.info(f"Predicting for time {prediction_timestamp}")
    prediction = pd.Series(
        ["D_17R_A_17L"] * len(lookaheads), index=lookaheads, name="config"
    )
    output_path = prediction_directory / prediction_timestamp.strftime(DATETIME_FORMAT)
    prediction.to_csv(output_path)


if __name__ == "__main__":
    typer.run(main)
