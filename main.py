import argparse
import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from thesis.analysis.breakpoints import calculate_break_points
from thesis.analysis.correlation import rolling_window
from thesis.config import DATA_PATH, OUTPUT_PATH
from thesis.lib.data.bond import BondDataLoader
from thesis.lib.data.fed_fund import FedFundDataLoader
from thesis.lib.data.inflation import InflationDataLoader
from thesis.lib.data.stock import StockDataLoader
from thesis.lib.data_processing import DataType
from thesis.lib.tests.bera import jarque_bera_test
from thesis.lib.tests.stationary import stationarity_test
from thesis.lib.visualization import plot_break_points, plot_df
from thesis.models.garch import Garch, GarchMethod
from thesis.utils.helpers import drop_nan_indexes, list_to_txt, select_common_indexes
from thesis.utils.logger import logger

cur_dir = Path(__file__).parent.resolve()

np.random.Generator(np.random.PCG64(42))


def parse_arguments() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Bond vs Stock Analysis Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-p", "--process-data", action="store_true", default=True, help="Process raw data and save cleaned versions"
    )
    parser.add_argument(
        "-P",
        "--skip-process-data",
        action="store_false",
        dest="process_data",
        help="Skip data processing, load from cache",
    )

    parser.add_argument(
        "-t",
        "--select-timeframe",
        action="store_true",
        default=False,
        help="Select specific timeframe (2000-01-01 to 2021-01-01)",
    )

    parser.add_argument("-c", "--create-plots", action="store_true", default=True, help="Create visualization plots")
    parser.add_argument("-C", "--skip-plots", action="store_false", dest="create_plots", help="Skip plot creation")

    parser.add_argument("-g", "--garch-model", action="store_true", default=True, help="Run GARCH model optimization")
    parser.add_argument(
        "-G", "--skip-garch", action="store_false", dest="garch_model", help="Skip GARCH model, load from cache"
    )

    parser.add_argument(
        "-s", "--test-stationary", action="store_true", default=False, help="Run stationarity tests on data"
    )

    parser.add_argument(
        "-j", "--test-jarque-bera", action="store_true", default=False, help="Run Jarque-Bera tests on data"
    )

    parser.add_argument(
        "-d",
        "--determine-correlation",
        action="store_true",
        default=True,
        help="Calculate rolling window and correlation matrix",
    )
    parser.add_argument(
        "-D",
        "--skip-correlation",
        action="store_false",
        dest="determine_correlation",
        help="Skip correlation analysis, load from cache",
    )

    parser.add_argument(
        "-b", "--calculate-breaks", action="store_true", default=True, help="Calculate breakpoints using kernel methods"
    )
    parser.add_argument(
        "-B", "--skip-breaks", action="store_false", dest="calculate_breaks", help="Skip breakpoint calculation"
    )

    parser.add_argument(
        "--start-date", type=str, default="2000-01-01", help="Start date for timeframe selection (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2021-01-01", help="End date for timeframe selection (YYYY-MM-DD)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    start_runtime = time.time()

    loader1 = BondDataLoader
    loader1_file = "DGS10 full.xls"
    loader1_data_type = DataType.XLSX
    loader2 = StockDataLoader
    loader2_file = "esindexfull.csv"
    loader2_data_type = DataType.CSV

    valid_bp_loader1 = FedFundDataLoader
    valid_bp_loader1_file = "FEDFUNDS.xls"
    valid_bp_loader1_data_type = DataType.XLSX
    valid_bp_loader1_column = "Rate"

    valid_bp_loader2 = InflationDataLoader
    valid_bp_loader2_file = "EXPINF10YR.xls"
    valid_bp_loader2_data_type = DataType.XLSX
    valid_bp_loader2_column = "Rate"

    rwc_days = [30, 60, 183, 365, 731, 1096]
    penalties_to_test = list(range(10, 100, 10))
    kernels_to_test = ["rbf", "linear", "cosine"]
    gammas_to_test = [50]

    folder_name = Path(f"{loader1.name}_vs_{loader2.name}")
    column_name = f"{loader1.name}_{loader2.name}"

    data_folder = cur_dir / OUTPUT_PATH / folder_name

    data_folder.mkdir(exist_ok=True, parents=True)
    full_data_path = cur_dir / DATA_PATH
    output_path = cur_dir / OUTPUT_PATH

    loaded_data_path = Path("loaded_data")
    common_data_path = Path("common_data")
    cleaned_data_path = Path("cleaned_data")

    if args.process_data:
        logger.info("Processing data...")
        loader1_path = full_data_path / loader1_file
        loader2_path = full_data_path / loader2_file
        loader1_data = loader1(loader1_path, loader1_data_type)
        loader2_data = loader2(loader2_path, loader2_data_type)

        loader1_data.save(loaded_data_path)
        loader2_data.save(loaded_data_path)

        common_indexes = select_common_indexes(loader1_data, loader2_data)

        list_to_txt(common_indexes, data_folder / "common_indexes")

        loader1_data.save(common_data_path)
        loader2_data.save(common_data_path)

        indexes_dropped = drop_nan_indexes(loader1_data, loader2_data)

        list_to_txt(indexes_dropped, data_folder / "indexes_dropped")

        loader1_data.save(cleaned_data_path)
        loader1_data.save(cleaned_data_path, DataType.PARQUET)
        loader2_data.save(cleaned_data_path)
        loader2_data.save(cleaned_data_path, DataType.PARQUET)

        if loader1_data.data is None or loader2_data.data is None:
            msg = "Data not loaded"
            raise ValueError(msg)

        valid_bp_loader1_path = full_data_path / valid_bp_loader1_file
        valid_bp_loader2_path = full_data_path / valid_bp_loader2_file
        valid_bp_loader1_data = valid_bp_loader1(valid_bp_loader1_path, valid_bp_loader1_data_type)
        valid_bp_loader2_data = valid_bp_loader2(valid_bp_loader2_path, valid_bp_loader2_data_type)

        valid_bp_loader1_data.save(loaded_data_path)
        valid_bp_loader2_data.save(loaded_data_path)

        if valid_bp_loader1_data.data is None or valid_bp_loader2_data.data is None:
            msg = "Validation data not loaded"
            raise ValueError(msg)

        first_date = min(loader1_data.data.index[0], loader2_data.data.index[0])
        last_date = max(loader1_data.data.index[-1], loader2_data.data.index[-1])

        first_date_valid1 = max(valid_bp_loader1_data.data.index[0], first_date)
        last_date_valid1 = min(valid_bp_loader1_data.data.index[-1], last_date)
        first_date_valid2 = max(valid_bp_loader2_data.data.index[0], first_date)
        last_date_valid2 = min(valid_bp_loader2_data.data.index[-1], last_date)

        valid_bp_loader1_data.data = valid_bp_loader1_data.data.loc[first_date_valid1:last_date_valid1]
        valid_bp_loader2_data.data = valid_bp_loader2_data.data.loc[first_date_valid2:last_date_valid2]

        valid_bp_loader1_data.save(cleaned_data_path)
        valid_bp_loader1_data.save(cleaned_data_path, DataType.PARQUET)
        valid_bp_loader2_data.save(cleaned_data_path)
        valid_bp_loader2_data.save(cleaned_data_path, DataType.PARQUET)
    else:
        logger.info("Loading data...")

        loader1_path = output_path / loader1.name / "cleaned_data.parquet"
        loader2_path = output_path / loader2.name / "cleaned_data.parquet"
        valid_bp_loader1_path = output_path / valid_bp_loader1.name / "cleaned_data.parquet"
        valid_bp_loader2_path = output_path / valid_bp_loader2.name / "cleaned_data.parquet"

        loader1_data = loader1(loader1_path, DataType.PARQUET)
        loader2_data = loader2(loader2_path, DataType.PARQUET)
        valid_bp_loader1_data = valid_bp_loader1(valid_bp_loader1_path, DataType.PARQUET)
        valid_bp_loader2_data = valid_bp_loader2(valid_bp_loader2_path, DataType.PARQUET)

    if loader1_data.data is None or loader2_data.data is None:
        msg = "Data not loaded"
        raise ValueError(msg)

    if valid_bp_loader1_data.data is None or valid_bp_loader2_data.data is None:
        msg = "Validation data not loaded"
        raise ValueError(msg)

    if args.select_timeframe:
        logger.info("Selecting specific timeframe...")
        start_datetime = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")  # noqa: DTZ007
        end_datetime = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")  # noqa: DTZ007

        loader1_data.data = loader1_data.data.loc[start_datetime:end_datetime]
        loader2_data.data = loader2_data.data.loc[start_datetime:end_datetime]

    returns = pd.concat(
        [loader1_data.data.loc[:, "Change_Demeaned"], loader2_data.data.loc[:, "Change_Demeaned"]], axis=1, join="inner"
    )
    returns.columns = [f"Change_Demeaned_{loader1_data.name}", f"Change_Demeaned_{loader2_data.name}"]

    if args.create_plots:
        logger.info("Creating plots...")
        loader1_data.plot(save_path="yield")
        loader1_data.plot(save_path="change", data_column="Change_Demeaned")
        loader1_data.plot_scatter(years_window=5, save_path="scatter")
        loader1_data.plot_scatter(years_window=0, save_path="scatter_all")
        loader2_data.plot(save_path="close")
        loader2_data.plot(save_path="change", data_column="Change_Demeaned")
        loader2_data.plot_scatter(years_window=5, save_path="scatter")
        loader2_data.plot_scatter(years_window=0, save_path="scatter_all")
        valid_bp_loader1_data.plot(save_path="fed_funds")
        valid_bp_loader2_data.plot(save_path="inflation")

    garch_epsilon_path = data_folder / "garch_epsilons.parquet"
    garch_sigma_path = data_folder / "garch_sigma.parquet"

    if args.garch_model:
        logger.info("Running GARCH model...")
        garch = Garch(returns, [loader1.name, loader2.name])
        optimized_params = garch.minimize()

        optimized_params_flat: list[Any] = [
            optimized_params[f"gamma_{loader1.name}"].iloc[0],
            optimized_params[f"gamma_{loader2.name}"].iloc[0],
            optimized_params[f"phi_{loader1.name}"].iloc[0],
            optimized_params[f"phi_{loader2.name}"].iloc[0],
            optimized_params[f"omega_{loader1.name}"].iloc[0],
            optimized_params[f"omega_{loader2.name}"].iloc[0],
            optimized_params[f"alpha_{loader1.name}"].iloc[0],
            optimized_params[f"alpha_{loader2.name}"].iloc[0],
            optimized_params[f"beta_{loader1.name}"].iloc[0],
            optimized_params[f"beta_{loader2.name}"].iloc[0],
        ]

        epsilon, sigma = garch.calculate(optimized_params_flat)

        epsilon.to_parquet(garch_epsilon_path)
        sigma.to_parquet(garch_sigma_path)
    else:
        logger.info("Loading GARCH model data...")

        if not garch_epsilon_path.is_file() or not garch_sigma_path.is_file():
            msg = "GARCH model data not found"
            raise FileNotFoundError(msg)

        epsilon = pd.read_parquet(garch_epsilon_path)
        sigma = pd.read_parquet(garch_sigma_path)

    if args.test_stationary:
        logger.info("Testing stationarity...")
        stationarity_test(loader1_data, "Change_Demeaned")
        # stationarity_test(bond_data, "Log_Returns_Change")  # noqa: ERA001
        stationarity_test(loader2_data, "Change_Demeaned")
        # stationarity_test(stock_data, "Log_Returns_Change")  # noqa: ERA001
        stationarity_test(epsilon, loader1.name)
        stationarity_test(epsilon, loader2.name)

    if args.test_jarque_bera:
        logger.info("Testing Jarque-Bera...")
        jarque_bera_test(loader1_data, "Change_Demeaned")
        jarque_bera_test(loader2_data, "Change_Demeaned")
        jarque_bera_test(epsilon, loader1.name)
        jarque_bera_test(epsilon, loader2.name)

    if args.determine_correlation:
        logger.info("Determining correlation...")
        rwcs: list[pd.DataFrame] = []
        rwc_path = data_folder / "rolling_window_correlation"
        rwc_path.mkdir(exist_ok=True)
        for rwc_day in rwc_days:
            rwc = rolling_window(loader1_data, loader2_data, f"{rwc_day}D", "Change_Demeaned", "Change_Demeaned")
            rwc *= -1
            rwc.columns = [column_name]
            rwc = rwc.iloc[2:, :]
            rwcs.append(rwc)

            rwc_file_path = rwc_path / f"{rwc_day}.parquet"

            rwc.to_parquet(rwc_file_path)
            nan_indexes = rwc.index[rwc.isna().any(axis=1)].astype(str).tolist()

            list_to_txt(nan_indexes, data_folder / f"correlation_nan_indexes_{rwc_day}")
            plot_df(
                rwc,
                None,
                f"Rolling window correlation {rwc_day} days",
                folder_name / "rolling_window_correlation" / f"{rwc_day}",
            )

        garch = Garch(returns, [loader1.name, loader2.name])

        method = GarchMethod.COVARIANCE
        optimized_alpha_beta = garch.minimize_correlation(sigma, method)

        alpha = optimized_alpha_beta[f"alpha_{method}"].iloc[0]
        beta = optimized_alpha_beta[f"beta_{method}"].iloc[0]

        correlation_matrix = garch.calculate_correlation_single(alpha, beta, sigma, method)
        correlation_matrix *= -1
        correlation_matrix = correlation_matrix.iloc[1:, :]

        plot_df(correlation_matrix, column_name, "Correlation matrix", folder_name / "correlation_matrix")
        correlation_matrix.to_parquet(data_folder / "correlation_matrix.parquet")
    else:
        logger.info("Loading correlation data...")

        if (
            not (data_folder / "rolling_window_correlation").is_dir()
            or not (data_folder / "correlation_matrix.parquet").is_file()
        ):
            msg = "Correlation data not found"
            raise FileNotFoundError(msg)

        rwcs = []
        for rwc_day in rwc_days:
            file_path = data_folder / "rolling_window_correlation" / f"{rwc_day}.parquet"
            if not file_path.is_file():
                msg = f"Rolling window correlation {rwc_day} not found"
                raise FileNotFoundError(msg)
            rwcs.append(pd.read_parquet(file_path))
        correlation_matrix = pd.read_parquet(data_folder / "correlation_matrix.parquet")

    # Select the common indexes
    common_indexes = rwcs[0].index.intersection(correlation_matrix.index).tolist()
    for idx, rwc in enumerate(rwcs):
        rwcs[idx] = rwc.loc[common_indexes]

    correlation_matrix = correlation_matrix.loc[common_indexes]

    break_points_folder = data_folder / "break_points"
    break_points_folder.mkdir(exist_ok=True)

    if args.calculate_breaks:
        logger.info("Calculating breaks...")
        for kernel in kernels_to_test:
            logger.info("Kernel: %s", kernel)
            for gamma in gammas_to_test:
                logger.info("Gamma: %s", gamma)
                params = {"gamma": gamma} if kernel == "rbf" else {}  # noqa: PLR2004

                for penalty in penalties_to_test:
                    logger.info("Penalty: %s", penalty)
                    for idx, rwc in enumerate(rwcs):
                        break_points_rwc = calculate_break_points(rwc, column_name, kernel, penalty, params)
                        break_points_rwc.to_parquet(
                            break_points_folder / f"rwc_{rwc_days[idx]}_{kernel}_{penalty}_{gamma}.parquet"
                        )
                        plot_break_points(
                            rwc,
                            column_name,
                            break_points_rwc,
                            "Rolling window correlation",
                            folder_name / f"rwc_{rwc_days[idx]}_{kernel}_{penalty}_{gamma}",
                        )

                    break_points_correlation = calculate_break_points(
                        correlation_matrix, column_name, kernel, penalty, params
                    )
                    break_points_correlation.to_parquet(
                        break_points_folder / f"correlation_{kernel}_{penalty}_{gamma}.parquet"
                    )
                    plot_break_points(
                        correlation_matrix,
                        column_name,
                        break_points_correlation,
                        "Correlation matrix",
                        folder_name / f"correlation_{kernel}_{penalty}_{gamma}",
                    )

                    break_points_validation1 = calculate_break_points(
                        valid_bp_loader1_data.data, valid_bp_loader1_column, kernel, penalty, params
                    )
                    break_points_validation1.to_parquet(
                        break_points_folder / f"{valid_bp_loader1.name}_{kernel}_{penalty}_{gamma}.parquet"
                    )
                    plot_break_points(
                        valid_bp_loader1_data.data,
                        valid_bp_loader1_column,
                        break_points_validation1,
                        valid_bp_loader1.name,
                        folder_name / f"{valid_bp_loader1.name}_{kernel}_{penalty}_{gamma}",
                    )

                    break_points_validation2 = calculate_break_points(
                        valid_bp_loader2_data.data, valid_bp_loader2_column, kernel, penalty, params
                    )
                    break_points_validation2.to_parquet(
                        break_points_folder / f"{valid_bp_loader2.name}_{kernel}_{penalty}_{gamma}.parquet"
                    )
                    plot_break_points(
                        valid_bp_loader2_data.data,
                        valid_bp_loader2_column,
                        break_points_validation2,
                        valid_bp_loader2.name,
                        folder_name / f"{valid_bp_loader2.name}_{kernel}_{penalty}_{gamma}",
                    )

    logger.info("Total runtime: %.2f seconds", time.time() - start_runtime)
