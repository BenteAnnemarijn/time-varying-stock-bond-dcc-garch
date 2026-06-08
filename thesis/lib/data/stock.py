import datetime
from pathlib import Path
from typing import ClassVar

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from thesis.config import PLOTS_PATH
from thesis.lib.data_processing import DataLoader, DataType
from thesis.lib.visualization import plot_df


class StockDataLoader(DataLoader):
    nan_check_columns: ClassVar[list[str]] = ["Close"]
    name = "stock"

    def process_data(self) -> None:
        """Process the stock data to calculate price changes and log returns."""
        if self.data_type != DataType.PARQUET:
            # Select relevant columns
            self.data = self.data.loc[:, ["OPEN", "HOOG", "LAAG", "SLOT", "fdatum"]]

            # Rename columns
            columns_to_rename = {"OPEN": "Open", "HOOG": "High", "LAAG": "Low", "SLOT": "Close", "fdatum": "Date"}
            self.data = self.data.rename(columns=columns_to_rename)

            # Convert date to datetime
            self.data["Date"] = pd.to_datetime(self.data["Date"])

            # Convert columns to float
            columns_to_float = ["Open", "High", "Low", "Close"]
            for column in columns_to_float:
                self.data[column] = self.data[column].astype(float)

            # set index
            self.data = self.data.set_index("Date")

            # Forward fill missing values to calculate percentage change and log returns
            self.data["Close_Fill"] = self.data["Close"].ffill()

            # Calculate percentage change
            self.data["Change"] = self.data["Close_Fill"].pct_change() * 100

            # Demeaned close change
            self.data["Change_Demeaned"] = self.data["Change"] - self.data["Change"].mean()

            # Calculate log returns
            self.data["Log_Returns"] = np.log(self.data["Close_Fill"] / self.data["Close_Fill"].shift(1))

            # Calculate log returns change
            self.data["Log_Returns_Change"] = self.data["Log_Returns"].pct_change() * 100

            self.data = self.data.drop(columns=["Close_Fill"])

            # drop first row as it contains the first NaN value
            self.data = self.data.iloc[1:]

    def plot(self, save_path: str | None = None, data_column: str = "Close") -> None:
        """Create a plot of the stock data."""
        if save_path:
            save_path = Path(self.name) / save_path
        plot_df(self.data, data_column, f"{data_column} prices SP500", save_path)

    def plot_scatter(self, years_window: int = 5, save_path: str | None = None) -> None:
        """Plot a scatter plot of the stock data.

        Args:
            years_window (int): The number of years to include in each scatter plot.
                If positive, it will plot the data for the last `years_window` years.
                If negative, it will raise a ValueError.

            save_path (Path | None): The path to save the plot. If None, it will display the plot instead.

        Raises:
            ValueError: If `years_window` is negative.

        """
        plt.style.use("seaborn-v0_8-darkgrid")
        if save_path:
            save_path = PLOTS_PATH / self.name / save_path

        if years_window > 0:
            start_year = int(min(self.data.index).year)
            end_year = int(max(self.data.index).year)

            for year in range(end_year - years_window, start_year, -years_window):
                start_date = datetime.datetime(year, 1, 1)  # noqa: DTZ001
                end_date = datetime.datetime(year + years_window, 1, 1)  # noqa: DTZ001

                data = self.data.loc[start_date:end_date]

                _, ax = plt.subplots(figsize=(12, 6))
                ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
                ax.scatter(data.index, data["Close"], label=f"{year}-{year + years_window}", s=1)
                ax.legend(frameon=False, fontsize=12)

                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Close", fontsize=14)
                plt.gcf().autofmt_xdate()
                plt.title(f"Closing prices SP500 from {year} to {year + years_window}", fontsize=16, pad=20)

                if save_path:
                    file_path = save_path.with_stem(f"{save_path.stem}_{year}-{year + years_window}")
                    file_path = file_path.with_suffix(".png")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(file_path)
                else:
                    plt.show()
                    input("Press Enter to continue...")
                plt.close()
        elif years_window < 0:
            msg = "Years window must be greater than 0"
            raise ValueError(msg)
        else:
            _, ax = plt.subplots(figsize=(12, 6))
            ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
            ax.scatter(self.data.index, self.data["Close"], label="Close", s=1)
            ax.legend(frameon=False, fontsize=12)

            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Close", fontsize=14)
            plt.title("Closing prices SP500 Scatter", fontsize=16, pad=20)
            plt.gcf().autofmt_xdate()

            if save_path:
                save_path = save_path.with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
            else:
                plt.show()
                input("Press Enter to continue...")

            plt.close()
