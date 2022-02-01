from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

feature_directory = Path("/codeexecution/data")
prediction_path = Path("/codeexecution/prediction.csv")


def read_airport_configs(airport_directory: Path) -> Tuple[str, pd.DataFrame]:
    """Reads the airport configuration features for a given airport data directory."""
    airport_code = airport_directory.name
    filename = f"{airport_code}_airport_config.csv"
    filepath = airport_directory / filename
    airport_config_df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return airport_code, airport_config_df


def make_prediction(
    airport_config_df_map: Dict[str, pd.DataFrame],
    pred_frame: pd.DataFrame,
    hedge: float = 1,
    weight: float = 6,
    discount_factor: float = 0.66,
) -> pd.Series:
    # start with a uniform distribution
    uniform = make_uniform(pred_frame) * hedge
    predictive_distribution = pd.DataFrame({"uniform": uniform})

    # select the data we're allowed to use
    first = pred_frame.iloc[0]
    airport_code, timestamp, lookahead, _, _ = first
    airport_config_df = airport_config_df_map[airport_code]
    current, subset = censor_data(airport_config_df, timestamp)

    # make the distribution of past configurations
    config_dist = make_config_dist(airport_code, subset, normalize=True)
    predictive_distribution["config_dist"] = config_dist.reindex(
        predictive_distribution.index
    ).fillna(0)
    other = config_dist.sum() - predictive_distribution.config_dist.sum()
    predictive_distribution.loc[f"{airport_code}:other", "config_dist"] += other

    # put some extra weight on the current configuration (or `other`)
    current_key = f"{airport_code}:{current}"
    if current_key not in pred_frame.config.values:
        current_key = f"{airport_code}:other"
    discount = 1 - discount_factor * (lookahead / 360)
    predictive_distribution["current"] = 0  # initalize a column of zeros
    predictive_distribution.loc[current_key, "current"] = weight * discount

    # combine the components and normalize the result
    mixture = predictive_distribution.sum(axis=1)
    predictive_distribution["mixture"] = mixture / mixture.sum()

    return predictive_distribution.mixture


def censor_data(
    airport_config_df: pd.DataFrame, timestamp: pd.Timestamp
) -> Tuple[str, pd.DataFrame]:
    mask = airport_config_df["timestamp"] <= timestamp
    subset = airport_config_df[mask]
    current = subset.iloc[-1].airport_config
    return current, subset


def make_all_predictions(
    airport_config_df_map: Dict[str, pd.DataFrame], predictions: pd.DataFrame
):
    """Predicts airport configuration for all of the prediction frames in a table."""
    all_preds = []
    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        pred_dist = make_prediction(
            airport_config_df_map, pred_frame, weight=100, hedge=10, discount_factor=0
        )
        assert np.array_equal(pred_dist.index.values, pred_frame["config"].values)
        all_preds.append(pred_dist.values)

    predictions["active"] = np.concatenate(all_preds)


def make_uniform(pred_frame: pd.DataFrame) -> pd.Series:
    indices = pred_frame["config"].values
    uniform = pd.Series(1, index=indices)
    uniform /= uniform.sum()
    return uniform


def make_config_dist(
    airport_code: str, airport_config_df: pd.DataFrame, normalize: bool = False
) -> pd.Series:
    config_timecourse = (
        airport_config_df.set_index("timestamp")
        .airport_config.resample("15min")
        .ffill()
        .dropna()
    )
    config_dist = config_timecourse.value_counts()
    if normalize:
        config_dist /= config_dist.sum()

    # prepend the airport code to the configuration strings
    prefix = pd.Series(f"{airport_code}:", index=config_dist.index)
    config_dist.index = prefix.str.cat(config_dist.index)
    return config_dist


def main(prediction_time: datetime):
    logger.info("Computing my predictions for {}", prediction_time)

    submission_format = pd.read_csv(
        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    )

    airport_directories = sorted(path for path in feature_directory.glob("k*"))

    airport_config_df_map = {}
    for airport_directory in sorted(airport_directories):
        airport_code, airport_config_df = read_airport_configs(airport_directory)
        print(airport_code)
        airport_config_df_map[airport_code] = airport_config_df

    submission_format = pd.read_csv(
        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    )
    print(f"{len(submission_format):,} rows x {len(submission_format.columns)} columns")

    submission = submission_format.copy().reset_index(drop=True)
    submission["active"] = np.nan

    make_all_predictions(airport_config_df_map, submission)

    submission.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)


if __name__ == "__main__":
    typer.run(main)
