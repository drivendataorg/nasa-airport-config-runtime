from datetime import datetime
from pathlib import Path
from loguru import logger
import pandas as pd
import typer

test_feature_directory = Path("/data")
user_feature_directory = Path("/codeexecution/data")
user_feature_directory.mkdir(exist_ok=True, parents=True)

airports = ["kdfw"]


def main(prediction_time: datetime):
    logger.info(f"Generating data subset for {prediction_time}")
    for airport in airports:
        test_airport_directory = test_feature_directory / airport
        user_airport_directory = user_feature_directory / airport
        user_airport_directory.mkdir(exist_ok=True, parents=True)
        for csv_path in test_airport_directory.glob("*.csv"):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df.loc[df.timestamp < prediction_time].to_csv(
                user_airport_directory / csv_path.name
            )


if __name__ == "__main__":
    typer.run(main)
