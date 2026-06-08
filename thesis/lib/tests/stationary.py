from lib.data_processing import DataLoader
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from thesis.utils.logger import logger


def stationarity_test(data: DataLoader | pd.DataFrame, column: str) -> None:
    """Check if the data is stationary.

    Args:
        data (DataLoader | pd.DataFrame): The data to test.
        column (str): The column to test for stationarity.

    Raises:
        ValueError: If the data is not loaded or the column does not exist.

    """
    if isinstance(data, pd.DataFrame):
        data_proces = data[column]
    else:
        if data.data is None:
            msg = "Data not loaded"
            raise ValueError(msg)

        data_proces = data.data[column]

    logger.info(f"Stationarity test for {column}")

    result = adfuller(data_proces)

    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")
    logger.info(f"Lags Used: {result[2]}")
    logger.info(f"Observations Used: {result[3]}")
    logger.info("Critical Values:")
    for key, value in result[4].items():
        logger.info(f"\t{key}: {value:.3f}")
