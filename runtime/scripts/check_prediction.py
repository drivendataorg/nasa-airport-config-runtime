from datetime import datetime
from pathlib import Path
import pandas as pd
import typer

prediction_directory = Path("/codeexecution/predictions")


def main(prediction_time: datetime):
    path = prediction_directory / f"{prediction_time}.csv"
    assert (
        path.exists()
    ), f"Submission did not generate prediction for {prediction_time}"
    prediction = pd.read_csv(prediction_directory / f"{prediction_time}.csv")
    assert prediction.columns == [
        "airport",
        "timestamp",
        "lookahead",
        "config",
        "probability",
    ]
    assert (
        prediction.groupby(["airport", "timestamp", "lookahead"]).probability.sum() == 1
    ).all(), "Probabilities for configurations at an airport, timestamp, lookahead do not sum to 1."


if __name__ == "__main__":
    typer.run(main)
