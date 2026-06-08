from typing import Any

import pandas as pd
from ruptures.detection import KernelCPD  # type: ignore[attr-defined]


def calculate_break_points(
    data: pd.DataFrame, column: str, kernel: str, penalty: int, params: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Calculate break points for a given column in the data using the specified kernel and penalty.

    Args:
        data (pd.DataFrame): The input data containing the column for which to calculate break points.
        column (str): The name of the column for which to calculate break points.
        kernel (str): The kernel to use for change point detection (e.g., "rbf", "linear").
        penalty (int): The penalty value to use for change point detection.
        params (dict[str, Any], optional): Additional parameters for the kernel. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the break points detected in the specified column.

    """
    data_process = data[column].to_numpy()

    model = KernelCPD(kernel=kernel, min_size=3, params=params).fit(data_process)  # type: ignore[no-untyped-call]

    breakpoints = model.predict(pen=penalty)  # type: ignore[no-untyped-call]

    if not breakpoints:
        return pd.DataFrame([], columns=["breakpoints"])

    return pd.DataFrame(breakpoints, columns=["breakpoints"])
