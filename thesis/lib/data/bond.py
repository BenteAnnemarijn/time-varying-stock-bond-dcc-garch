import datetime
from pathlib import Path
from typing import ClassVar

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from thesis.config import PLOTS_PATH
from thesis.lib.data_processing import DataLoader, DataType
from thesis.lib.visualization import plot_df


class BondDataLoader(DataLoader):
    nan_check_columns: ClassVar[list[str]] = ["Yield"]
    name = "bond"

    def process_data(self) -> None:
        """Process the bond data to calculate yield changes and log returns."""
        if self.data_type != DataType.PARQUET:
            # Remove the first 10 rows as they contain introductory text
            self.data = self.data.iloc[10:]

            # Rename columns
            self.data.columns = ["Date", "Yield"]

            # Convert date to datetime
            self.data["Date"] = pd.to_datetime(self.data["Date"])

            # Convert yield to float
            self.data["Yield"] = self.data["Yield"].astype(float)

            # Create forward fill for missing values to calculate percentage change
            self.data["Yield_Fill"] = self.data["Yield"].ffill()

            # set index
            self.data = self.data.set_index("Date")

            # Calculate percentage change
            self.data["Change"] = self.data["Yield_Fill"].pct_change() * 100

            # Demeaned yield change
            self.data["Change_Demeaned"] = self.data["Change"] - self.data["Change"].mean()

            # Calculate log returns
            self.data["Log_Returns"] = np.log(self.data["Yield_Fill"] / self.data["Yield_Fill"].shift(1))

            # Calculate log returns change
            self.data["Log_Returns_Change"] = self.data["Log_Returns"].pct_change() * 100

            self.data = self.data.drop(columns=["Yield_Fill"])

            # drop first row as it contains the first NaN value
            self.data = self.data.iloc[1:]

    def plot(self, save_path: str | None = None, data_column: str = "Yield") -> None:
        """Create a plot of the bond data."""
        if save_path:
            save_path = Path(self.name) / save_path
        plot_df(self.data, data_column, f"Daily 10-year Treasury {data_column}", save_path)

    def plot_scatter(self, years_window: int = 5, save_path: str | None = None) -> None:
        """Plot a scatter plot of the bond data.

        Args:
            years_window (int): The number of years to plot in the scatter plot. If
                positive, it will plot the data for the last `years_window` years.
                If negative, it will raise a ValueError.
            save_path (Path | None): The path to save the plot. If None, it
                will display the plot instead.

        Raises:
            ValueError: If `years_window` is negative.

        """
        plt.style.use("seaborn-v0_8-darkgrid")
        if save_path:
            save_path = Path(PLOTS_PATH) / self.name / save_path

        if years_window > 0:
            start_year = int(min(self.data.index).year)
            end_year = int(max(self.data.index).year)

            for year in range(end_year - years_window, start_year, -years_window):
                start_date = datetime.datetime(year, 1, 1)  # noqa: DTZ001
                end_date = datetime.datetime(year + years_window, 1, 1)  # noqa: DTZ001

                data = self.data.loc[start_date:end_date]

                _, ax = plt.subplots(figsize=(12, 6))
                ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
                ax.scatter(data.index, data["Yield"], label=f"{year}-{year + years_window}", s=1)
                ax.legend(frameon=False, fontsize=12)

                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Yield", fontsize=14)
                plt.gcf().autofmt_xdate()
                plt.title(f"Daily 10-year Treasury Yield from {year} to {year + years_window}", fontsize=16, pad=20)

                if save_path:
                    save_path = save_path.with_suffix("")
                    save_path = save_path.with_name(f"{save_path.name}_{year}-{year + years_window}.png")

                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path)
                else:
                    plt.show()
                    input("Press Enter to continue...")
                plt.close()
        elif years_window < 0:
            msg = "Years window must be positive"
            raise ValueError(msg)
        else:
            _, ax = plt.subplots(figsize=(12, 6))
            ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
            ax.scatter(self.data.index, self.data["Yield"], label="Yield", s=1)
            ax.legend(frameon=False, fontsize=12)

            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Yield", fontsize=14)
            plt.gcf().autofmt_xdate()
            plt.title("Daily 10-year Treasury Yield Scatter", fontsize=16, pad=20)

            if save_path:
                save_path = save_path.with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
            else:
                plt.show()
                input("Press Enter to continue...")
            plt.close()
