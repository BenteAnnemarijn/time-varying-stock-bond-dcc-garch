from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import pandas as pd

from thesis.config import OUTPUT_PATH


class DataType(StrEnum):
    XLSX = ".xlsx"
    CSV = ".csv"
    JSON = ".json"
    XML = ".xml"
    FEATHER = ".feather"
    PARQUET = ".parquet"


ALLOWED_XLSX_EXTENSIONS = {DataType.XLSX.value, ".xls"}


class DataLoader(ABC):
    nan_check_columns: ClassVar[list[str]] = []
    default_output_type = DataType.XLSX
    name = "DataLoader"

    def __init__(self, data_path: Path, data_type: DataType) -> None:
        if not data_path.is_file():
            data_path.parent.mkdir(parents=True, exist_ok=True)

            msg = f"File {data_path} not found. Please provide the file."
            raise FileNotFoundError(msg)

        self.data_path = data_path
        self.data_type = data_type
        self.data: pd.DataFrame | None = None

        self.load_data()

    def load_data(self) -> None:
        """Load data from the specified path based on the data type."""
        if self.data_type == DataType.XLSX:
            self.__load_xlsx()
        elif self.data_type == DataType.CSV:
            self.__load_csv()
        elif self.data_type == DataType.JSON:
            self.__load_json()
        elif self.data_type == DataType.XML:
            self.__load_xml()
        elif self.data_type == DataType.FEATHER:
            self.__load_feather()
        elif self.data_type == DataType.PARQUET:
            self.__load_parquet()

        self.process_data()

    def save(self, output_path: Path, data_type: DataType | None = None) -> None:
        """Save the processed data to the specified path in the given format."""
        if not data_type:
            data_type = self.default_output_type

        output_path = OUTPUT_PATH / self.name / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data_type == DataType.XLSX:
            self.__save_xlsx(output_path)
        elif data_type == DataType.CSV:
            self.__save_csv(output_path)
        elif data_type == DataType.JSON:
            self.__save_json(output_path)
        elif data_type == DataType.XML:
            self.__save_xml(output_path)
        elif data_type == DataType.FEATHER:
            self.__save_feather(output_path)
        elif data_type == DataType.PARQUET:
            self.__save_parquet(output_path)

    @abstractmethod
    def process_data(self) -> None:
        """Process the data loaded from the file."""

    @abstractmethod
    def plot(self, save_path: Path | None = None) -> None:
        """Create a plot of the data."""

    def get_nan_columns_to_check(self) -> pd.DataFrame:
        """Get the columns to check for NaN values.

        Returns:
            pd.DataFrame: DataFrame containing the columns to check for NaN values.

        Raises:
            ValueError: If the data is not loaded or the columns to check are not defined.

        """
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        return self.data.loc[:, self.nan_check_columns]

    def get_nan_indexes(self) -> list[str]:
        """Get the indexes of the rows with NaN values in the specified columns.

        Returns:
            list[str]: List of indexes with NaN values in the specified columns.

        Raises:
            ValueError: If the data is not loaded or the columns to check are not defined.

        """
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        return self.data.index[self.get_nan_columns_to_check().isna().any(axis=1)].astype(str).tolist()

    def get_data_without_nan(self) -> pd.DataFrame:
        """Get the data without rows containing NaN values in the specified columns.

        Returns:
            pd.DataFrame: DataFrame without rows containing NaN values in the specified columns.

        Raises:
            ValueError: If the data is not loaded or the columns to check are not defined.

        """
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        data_copy = self.data.copy(deep=True)
        return data_copy.drop(self.get_nan_indexes())

    def __load_xlsx(self) -> None:
        if self.data_path.suffix not in ALLOWED_XLSX_EXTENSIONS:
            msg = "File is not an xlsx file"
            raise ValueError(msg)
        self.data = pd.read_excel(self.data_path)

    def __load_csv(self) -> None:
        if self.data_path.suffix != DataType.CSV.value:
            msg = "File is not a csv file"
            raise ValueError(msg)
        self.data = pd.read_csv(self.data_path, low_memory=False)

    def __load_json(self) -> None:
        if self.data_path.suffix != DataType.JSON.value:
            msg = "File is not a json file"
            raise ValueError(msg)
        self.data = pd.read_json(self.data_path)

    def __load_xml(self) -> None:
        if self.data_path.suffix != DataType.XML.value:
            msg = "File is not an xml file"
            raise ValueError(msg)
        self.data = pd.read_xml(self.data_path)

    def __load_feather(self) -> None:
        if self.data_path.suffix != DataType.FEATHER.value:
            msg = "File is not a feather file"
            raise ValueError(msg)
        self.data = pd.read_feather(self.data_path)

    def __load_parquet(self) -> None:
        if self.data_path.suffix != DataType.PARQUET.value:
            msg = "File is not a parquet file"
            raise ValueError(msg)
        self.data = pd.read_parquet(self.data_path)

    def __save_xlsx(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.XLSX.value:
            path = path.with_suffix(DataType.XLSX.value)
        self.data.to_excel(path)

    def __save_csv(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.CSV.value:
            path = path.with_suffix(DataType.CSV.value)
        self.data.to_csv(path)

    def __save_json(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.JSON.value:
            path = path.with_suffix(DataType.JSON.value)
        self.data.to_json(path)

    def __save_xml(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.XML.value:
            path = path.with_suffix(DataType.XML.value)
        self.data.to_xml(path)

    def __save_feather(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.FEATHER.value:
            path = path.with_suffix(DataType.FEATHER.value)
        self.data.to_feather(path)

    def __save_parquet(self, path: Path) -> None:
        if self.data is None:
            msg = "Data is empty"
            raise ValueError(msg)
        if path.suffix != DataType.PARQUET.value:
            path = path.with_suffix(DataType.PARQUET.value)
        self.data.to_parquet(path)

    def __str__(self) -> str:
        return f"{self.name}"
