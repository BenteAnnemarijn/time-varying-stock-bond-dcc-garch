import os
from pathlib import Path
from typing import Any

from thesis.lib.data_processing import DataLoader

PLOTS_PATH = os.getenv("PLOTS_PATH", "data/output/plots")


def select_common_indexes(data1: DataLoader, data2: DataLoader) -> list[str]:
    """Select the common indexes from two dataframes.

    Args:
        data1 (DataLoader): The first DataLoader instance containing the first dataframe.
        data2 (DataLoader): The second DataLoader instance containing the second dataframe.

    Returns:
        list[str]: A list of common indexes between the two dataframes.

    Raises:
        ValueError: If the data in either DataLoader instance is not loaded.

    """
    if data1.data is None or data2.data is None:
        msg = "Data not loaded"
        raise ValueError(msg)

    common_indexes = data1.data.index.intersection(data2.data.index)
    data1.data = data1.data.loc[common_indexes]
    data2.data = data2.data.loc[common_indexes]

    return common_indexes.astype(str).tolist()


def drop_nan_indexes(data1: DataLoader, data2: DataLoader) -> list[str]:
    """Drop the indexes where both dataframes have NaN values.

    Args:
        data1 (DataLoader): The first DataLoader instance containing the first dataframe.
        data2 (DataLoader): The second DataLoader instance containing the second dataframe.

    Returns:
        list[str]: A list of indexes that were dropped due to NaN values in either dataframe

    Raises:
        ValueError: If the data in either DataLoader instance is not loaded.

    """
    if data1.data is None or data2.data is None:
        msg = "Data not loaded"
        raise ValueError(msg)

    nan_indexes1 = data1.data.index[data1.get_nan_columns_to_check().isna().any(axis=1)]
    nan_indexes2 = data2.data.index[data2.get_nan_columns_to_check().isna().any(axis=1)]
    nan_indexes = nan_indexes1.union(nan_indexes2)

    data1.data = data1.data.drop(nan_indexes)
    data2.data = data2.data.drop(nan_indexes)

    return nan_indexes.astype(str).tolist()


def list_to_txt(data: list[Any], path: Path) -> None:
    """Write a list of data to a text file.

    Args:
        data (list[Any]): The list of data to write to the text file.
        path (Path): The path to the text file where the data should be written.

    """
    path = path.with_suffix(".txt")

    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")
