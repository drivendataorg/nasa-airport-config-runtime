from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
runtime_directory = Path(__file__).parents[1]

app = typer.Typer(add_completion=False)


def create_fake_features(
    column_values: dict,
    n: int,
    start_time: datetime,
    end_time: datetime,
    output_path: Path,
):
    df = pd.DataFrame(
        {
            column: np.random.choice(column_values[column], size=n)
            for column in column_values
        }
    )
    df["timestamp"] = pd.date_range(start_time, end_time, periods=len(df))
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False, date_format=DATETIME_FORMAT)


@app.command()
def main(
    submission_format_path: Path = runtime_directory
    / "scripts"
    / "submission_format.csv.bz2",
    fake_data_params_path: Path = runtime_directory
    / "scripts"
    / "fake_data_params.json",
    output_directory: Path = runtime_directory / "data",
):
    """Generates a small, not very realistic data that nonetheless can stand in as the features and
    submission format for testing out the code execution submission.

    Reads fake data parameters where each feature CSV is described by the column names and 5 sample
    values for each column. 100 rows are simulated by sampling (with replacement) from those sample
    values. The start time for the fake data is the first time in the submission format, and the end
    time is the last time in the submission format.
    """
    submission_format = pd.read_csv(submission_format_path, parse_dates=["timestamp"])
    start_time = submission_format.iloc[0].timestamp
    end_time = submission_format.iloc[-1].timestamp

    with Path(fake_data_params_path).open("r") as fp:
        fakes = json.load(fp)

    for fake in fakes:
        create_fake_features(
            fake["values"],
            n=100,
            start_time=start_time,
            end_time=end_time,
            output_path=output_directory / fake["relative_path"],
        )

    submission_format.to_csv(
        output_directory / "submission_format.csv", date_format=DATETIME_FORMAT
    )


if __name__ == "__main__":
    app()
