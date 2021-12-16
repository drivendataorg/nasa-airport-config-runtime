from datetime import datetime
from pathlib import Path
from loguru import logger
import pandas as pd
import typer

test_feature_directory = Path("/data/test_features")
user_feature_directory = Path("/codeexecution/data")
user_feature_directory.mkdir(exist_ok=True, parents=True)


def main(prediction_time: datetime):
    logger.info(f"Generating data subset for {prediction_time}")
    for path in test_feature_directory.glob("*"):
        if path.is_dir():
            airport = path.name
            airport_directory = user_feature_directory / airport
            airport_directory.mkdir(exist_ok=True, parents=True)
            for csv_path in path.glob("*.csv"):
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                df.loc[df.timestamp < prediction_time].to_csv(
                    airport_directory / csv_path.name
                )


if __name__ == "__main__":
    typer.run(main)
