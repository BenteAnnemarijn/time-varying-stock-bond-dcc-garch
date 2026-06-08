from pathlib import Path
from typing import ClassVar

import pandas as pd

from thesis.lib.data_processing import DataLoader, DataType
from thesis.lib.visualization import plot_df


class InflationDataLoader(DataLoader):
    nan_check_columns: ClassVar[list[str]] = ["Rate"]
    name = "inflation"

    def process_data(self) -> None:
        """Process the inflation data to calculate rate changes and log returns."""
        if self.data_type != DataType.PARQUET:
            # Remove the first 10 rows as they contain introductory text
            self.data = self.data.iloc[10:]

            # Rename columns
            self.data.columns = ["Date", "Rate"]

            # Convert date to datetime
            self.data["Date"] = pd.to_datetime(self.data["Date"])

            # Convert Rate to float
            self.data["Rate"] = self.data["Rate"].astype(float)

            # set index
            self.data = self.data.set_index("Date")

            # Create forward fill for missing values to calculate percentage change
            self.data["Rate_Fill"] = self.data["Rate"].ffill()

            # Calculate percentage change
            self.data["Change"] = self.data["Rate_Fill"].pct_change() * 100

            # Demeaned Rate change
            self.data["Change_Demeaned"] = self.data["Change"] - self.data["Change"].mean()

            self.data = self.data.drop(columns=["Rate_Fill"])

            # drop first row as it contains the first NaN value
            self.data = self.data.iloc[1:]

    def plot(self, save_path: str | None = None, data_column: str = "Rate") -> None:
        """Create a plot of the inflation data."""
        if save_path:
            save_path = Path(self.name) / save_path
        plot_df(self.data, data_column, "Daily Rate", save_path)
