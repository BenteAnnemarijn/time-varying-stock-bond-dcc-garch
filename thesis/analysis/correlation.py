import pandas as pd

from thesis.lib.data_processing import DataLoader
from thesis.utils.logger import logger


def rolling_window(
    base_data: DataLoader, correlation_data: DataLoader, window_size: str, base_column: str, correlation_column: str
) -> pd.DataFrame:
    """Calculate the rolling window correlation between two columns from two different DataLoader instances.

    Args:
        base_data (DataLoader): The first DataLoader instance containing the base data.
        correlation_data (DataLoader): The second DataLoader instance containing the data to correlate with the base data.
        window_size (str): The size of the rolling window (e.g., "30D" for 30 days).
        base_column (str): The name of the column in the base_data to use for correlation.
        correlation_column (str): The name of the column in the correlation_data to use for correlation.

    Returns:
        pd.DataFrame: A DataFrame containing the rolling window correlation values.

    Raises:
        ValueError: If the data in either DataLoader instance is not loaded.

    """
    if base_data.data is None or correlation_data.data is None:
        msg = "Data not loaded"
        raise ValueError(msg)

    base_data_data = base_data.data[base_column]
    correlation_data_data = correlation_data.data[correlation_column]

    rolling_correlation = base_data_data.rolling(window=window_size).corr(correlation_data_data)

    logger.info("Rolling Window Correlation")
    logger.info(f"Number of NaNs: {rolling_correlation.isna().sum()}")
    logger.info(f"Number of values: {len(rolling_correlation)}")
    logger.info(f"Percentage of NaNs: {rolling_correlation.isna().sum() / len(rolling_correlation) * 100}")

    return rolling_correlation.to_frame()
