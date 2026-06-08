from lib.data_processing import DataLoader
import pandas as pd
from scipy.stats import jarque_bera

from thesis.utils.logger import logger


def jarque_bera_test(data: DataLoader | pd.DataFrame, column: str) -> None:
    """Check if the data is normally distributed.

    Args:
        data (DataLoader | pd.DataFrame): The data to test.
        column (str): The column to test for normality.

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

    logger.info(f"Jarque-Bera test for {column}")

    result = jarque_bera(data_proces, nan_policy="omit")

    logger.info("Jarque-Bera Results:", result)
