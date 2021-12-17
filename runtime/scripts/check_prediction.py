from datetime import datetime

from loguru import logger
import numpy as np
import pandas as pd
import typer
from pathlib import Path

DATETIME_FORMAT = "%Y-%m-%dT%H:%M"
prediction_directory = Path("/codeexecution/predictions")
expected_columns = [
    "airport",
    "timestamp",
    "lookahead",
    "config",
    "probability",
]


def main(prediction_time: datetime):
    """Checks the predictions for a single timepoint to make sure the expected columns are present
    and the probabiilties for configurations at a single airport, timestamp, and lookahead sum to 1.
    """
    path = prediction_directory / f"{prediction_time.strftime(DATETIME_FORMAT)}.csv"
    assert (
        path.exists()
    ), f"Submission did not generate prediction for {prediction_time}"

    prediction = pd.read_csv(path)

    if (len(prediction.columns) != len(expected_columns)) or (
        prediction.columns.tolist() != expected_columns
    ):
        raise ValueError(
            f"""Prediction columns {", ".join(prediction.columns)} do not match expected columns """
            f"""{", ".join(expected_columns)}"""
        )

    airport_config_probability_sums = prediction.groupby(
        ["airport", "timestamp", "lookahead"]
    ).probability.sum()
    if not np.allclose(airport_config_probability_sums, 1):
        logger.error(prediction)
        logger.error(airport_config_probability_sums)
        raise ValueError(
            "Probabilities for configurations at an airport, timestamp, lookahead do not sum to 1."
        )


if __name__ == "__main__":
    typer.run(main)
