from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from thesis.lib.data_processing import DataLoader, DataType
from thesis.lib.visualization import plot_df


class FedFundDataLoader(DataLoader):
    nan_check_columns: ClassVar[list[str]] = ["Rate"]
    name = "fed_fund"

    def process_data(self) -> None:
        """Process the Federal Fund Rate data to calculate rate changes and log returns."""
        if self.data_type != DataType.PARQUET:
            # Remove the first 10 rows as they contain introductory text
            self.data = self.data.iloc[10:]

            # Rename columns
            self.data.columns = ["Date", "Rate"]

            # Convert date to datetime
            self.data["Date"] = pd.to_datetime(self.data["Date"])

            # Convert rate to float
            self.data["Rate"] = self.data["Rate"].astype(float)

            # set index
            self.data = self.data.set_index("Date")

            self.data["Rate_Fill"] = self.data["Rate"].ffill()

            # Calculate percentage change
            self.data["Change"] = self.data["Rate_Fill"].pct_change() * 100

            # Demeaned rate change
            self.data["Change_Demeaned"] = self.data["Change"] - self.data["Change"].mean()

            # Calculate log returns
            self.data["Log_Returns"] = np.log(self.data["Rate_Fill"] / self.data["Rate_Fill"].shift(1))

            # Calculate log returns change
            self.data["Log_Returns_Change"] = self.data["Log_Returns"].pct_change() * 100

            self.data = self.data.drop(columns=["Rate_Fill"])

            # drop first row as it contains the first NaN value
            self.data = self.data.iloc[1:]

    def plot(self, save_path: str | None = None, data_column: str = "Rate") -> None:
        """Create a plot of the Federal Fund Rate data."""
        if save_path:
            save_path = Path(self.name) / save_path
        plot_df(self.data, data_column, "Daily Federal Fund Rate", save_path)
