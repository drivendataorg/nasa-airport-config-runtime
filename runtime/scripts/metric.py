from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import typer


def main(predictions_path: Path, labels_path: Path):
    """Computes the mean-aggregated log loss for a set of predictions given some test labels."""
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    score = np.mean(
        [
            log_loss(
                labels.loc[labels.airport == airport],
                predictions.loc[predictions.airport == airport],
            )
            for airport in labels.airports.unique()
        ]
    )
    logger.success(f"Score: {score}")


if __name__ == "__main__":
    typer.run(main)
