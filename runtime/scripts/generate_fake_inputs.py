from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def main(input_dir: Path, output_dir: Path, seed: int = 79):
    rng = np.random.RandomState(seed)
    output_dir.mkdir(exist_ok=True, parents=True)
    min_timestamp = pd.Timestamp.max
    max_timestamp = pd.Timestamp.min
    for path in Path(input_dir).glob("*"):
        logger.info(f"Processing {path.name}")
        if path.is_dir():
            airport = path.name
            for csv_path in path.rglob("*.csv"):
                logger.info(f"Creating sample of {csv_path.name}")
                (output_dir / airport).mkdir(exist_ok=True, parents=True)
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                min_timestamp = min(min_timestamp, df.timestamp.min())
                max_timestamp = max(max_timestamp, df.timestamp.max())
                df.sample(50, random_state=rng).to_csv(
                    output_dir / airport / csv_path.name, index=False
                )
        else:
            logger.info(f"Creating sample of {path.name}")
            pd.read_csv(path).to_csv(output_dir / path.name, index=False)

    min_timestamp, max_timestamp = min_timestamp.floor("1H"), max_timestamp.ceil("1H")
    prediction_times = pd.date_range(min_timestamp, max_timestamp, freq="1H")
    logger.info(
        f"Creating {len(prediction_times)} prediction times between {min_timestamp} and {max_timestamp}"
    )
    prediction_times.to_series().dt.strftime(DATETIME_FORMAT).reset_index(
        drop=True
    ).to_csv(output_dir / "prediction_times.txt", index=False, header=None)


if __name__ == "__main__":
    typer.run(main)
