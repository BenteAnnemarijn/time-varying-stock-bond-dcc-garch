from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from thesis.config import OUTPUT_PATH
from thesis.utils.logger import logger


class GarchMethod(StrEnum):
    CORRELATION = "correlation"
    COVARIANCE = "covariance"
    MEAN = "mean"
    EWMA = "ewma"


def safe_log_det(matrix: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate the safe logarithm of the determinant of a matrix.

    If the determinant is non-positive, return the logarithm of a small positive value (epsilon).

    Args:
        matrix (np.ndarray): The input matrix for which to calculate the log determinant.
        epsilon (float): A small positive value to use when the determinant is non-positive.

    Returns:
        float: The logarithm of the determinant of the matrix, or the logarithm of epsilon

    """
    det = np.linalg.det(matrix)
    if det <= 0:
        return float(np.log(epsilon))
    return float(np.log(det))


def regularize_matrix(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Add a small value to the diagonal of the matrix to ensure it is invertible.

    Args:
        matrix (np.ndarray): The input matrix to regularize.
        epsilon (float): A small value to add to the diagonal elements of the matrix.

    Returns:
        np.ndarray: The regularized matrix with a small value added to the diagonal.

    """
    return matrix + np.eye(matrix.shape[0]) * epsilon


class Garch:
    iteration = 0
    gamma_init = 0.1
    phi_init = 0.1
    omega_init = 0.01
    alpha_init = 0.1
    beta_init = 0.8

    gamma_bounds_init = (-10e10, 10e10)
    phi_bounds_init = (-1, 1)
    omega_bounds_init = (1e-10, 10e10)
    alpha_bounds_init = (1e-10, 1 - 1e-10)
    beta_bounds_init = (1e-10, 1 - 1e-10)

    def __init__(self, data: pd.DataFrame, names: list[str]) -> None:
        self.data = data
        self.column_count = self.data.shape[1]
        self.row_count = self.data.shape[0]
        self.names = names

        self.reversed_shape = (self.column_count, self.row_count)

        self.gamma = [self.gamma_init] * self.column_count
        self.phi = [self.phi_init] * self.column_count
        self.omega = [self.omega_init] * self.column_count
        self.alpha = [self.alpha_init] * self.column_count
        self.beta = [self.beta_init] * self.column_count

        self.gamma_bounds = [self.gamma_bounds_init] * self.column_count
        self.phi_bounds = [self.phi_bounds_init] * self.column_count
        self.omega_bounds = [self.omega_bounds_init] * self.column_count
        self.alpha_bounds = [self.alpha_bounds_init] * self.column_count
        self.beta_bounds = [self.beta_bounds_init] * self.column_count

    @staticmethod
    def calculate_individual(params: list[Any], data: pd.DataFrame) -> float:
        """Calculate the log-likelihood of the GARCH model for the given parameters and data.

        Args:
            params (list[Any]): A list of parameters for the GARCH model, including gamma, phi, omega, alpha, and beta.
            data (pd.DataFrame): The input data for which to calculate the log-likelihood.

        Returns:
            float: The log-likelihood of the GARCH model for the given parameters and data.

        """
        rows = data.shape[0]
        columns = data.shape[1]
        reversed_shape = (columns, rows)
        data_values = data.to_numpy()

        gamma = params[:columns]
        phi = params[columns : columns * 2]
        omega = params[columns * 2 : columns * 3]
        alpha = params[columns * 3 : columns * 4]
        beta = params[columns * 4 :]

        epsilons = np.zeros(reversed_shape)
        sigma = np.zeros(reversed_shape)
        log_likelihood: float = 0.0

        for i in range(columns):
            epsilons[(i, 0)] = data_values[(1, i)] - (gamma[i] + phi[i] * data_values[(0, i)])
            sigma[(i, 0)] = omega[i]

        for i in range(columns):
            for t in range(1, rows):
                sigma[(i, t)] = omega[i] + alpha[i] * epsilons[(i, t - 1)] ** 2 + beta[i] * sigma[(i, t - 1)]
                epsilons[(i, t)] = data_values[(t, i)] - (gamma[i] + phi[i] * data_values[(t - 1, i)])
                log_likelihood += np.log(2 * np.pi) + np.log(sigma[(i, t)]) + (data_values[(t, i)] ** 2 / sigma[(i, t)])

        log_likelihood /= 2

        return log_likelihood

    @staticmethod
    def calculate_correlation(  # noqa: PLR0914
        params: list[Any], data: pd.DataFrame, sigma: pd.DataFrame, method: GarchMethod = GarchMethod.CORRELATION
    ) -> float:
        """Calculate the log-likelihood of the GARCH model with correlation for the given parameters, data, and sigma.

        Args:
            params (list[Any]): A list of parameters for the GARCH model, including alpha and beta.
            data (pd.DataFrame): The input data for which to calculate the log-likelihood.
            sigma (pd.DataFrame): The conditional variances for each time step and variable.
            method (str): The method to use for calculating the correlation matrix. Options are "correlation", "covariance", "mean", and "ewma".

        Returns:
            float: The log-likelihood of the GARCH model with correlation for the given parameters, data, and sigma.

        """
        rows = data.shape[0]

        alpha, beta = params
        log_likelihood: float = 0.0

        data_array = (data.to_numpy() / np.sqrt(sigma.to_numpy())).T

        if method == GarchMethod.CORRELATION:
            q_bar_matrix = np.corrcoef(data_array)
        elif method == GarchMethod.COVARIANCE:
            q_bar_matrix = np.cov(data_array)
        elif method == GarchMethod.MEAN:
            mean_returns = data_array.mean(axis=1)
            q_bar_matrix = np.outer(mean_returns, mean_returns)
        elif method == GarchMethod.EWMA:
            # Calculate exponentially weighted moving average covariance matrix
            # Assuming a decay factor (lambda) of 0.94, which is common for financial data
            decay_factor = 0.94
            q_bar_matrix = pd.DataFrame(data_array).ewm(alpha=1 - decay_factor).cov().to_numpy()[-rows:]

        covariance_matrix = [q_bar_matrix * (1 - alpha - beta)]
        correlation_matrix_list: list[np.ndarray] = []

        for t in range(1, rows + 1):
            z_vector = data_array[:, t - 1].reshape(-1, 1)
            z_matrix = z_vector @ z_vector.T

            covariance_matrix.append(
                q_bar_matrix * (1 - alpha - beta) + alpha * z_matrix + beta * covariance_matrix[t - 1]
            )
            diag_covariance = np.diag(np.diag(covariance_matrix[t]))

            correlation_matrix_list.append(
                np.linalg.inv(np.sqrt(diag_covariance)) @ covariance_matrix[t] @ np.linalg.inv(np.sqrt(diag_covariance))
            )

        correlation_matrix: np.ndarray = np.array(correlation_matrix_list)

        for t in range(2, rows):
            correlation_vector = correlation_matrix[t - 1]
            z_vector = data_array[:, t].reshape(-1, 1)
            log_det_r_t = np.log(np.linalg.det(correlation_vector))
            zrz = z_vector.T @ np.linalg.inv(correlation_vector) @ z_vector
            log_likelihood += log_det_r_t + zrz - (z_vector.T @ z_vector)

        log_likelihood = np.sum(log_likelihood) / 2

        logger.info(f"Alpha: {alpha}, Beta: {beta}, Log-likelihood: {log_likelihood}")

        return log_likelihood

    def calculate_correlation_single(  # noqa: PLR0914
        self, alpha: float, beta: float, sigma: pd.DataFrame, method: GarchMethod = GarchMethod.CORRELATION
    ) -> pd.DataFrame:
        """Calculate the correlation matrix for a single set of alpha and beta parameters.

        Args:
            alpha (float): The alpha parameter for the GARCH model.
            beta (float): The beta parameter for the GARCH model.
            sigma (pd.DataFrame): The conditional variances for each time step and variable.
            method (GarchMethod): The method to use for calculating the correlation matrix. Options are "correlation", "covariance", "mean", and "ewma".

        Returns:
            pd.DataFrame: A DataFrame containing the correlation matrix for each time step.

        """
        rows = self.data.shape[0] - 1

        data_process = self.data.iloc[1:, :]
        sigma_process = sigma.iloc[1:, :]

        data_array = (data_process.to_numpy() / np.sqrt(sigma_process.to_numpy())).T

        if method == GarchMethod.CORRELATION:
            q_bar_matrix = np.corrcoef(data_array)
        elif method == GarchMethod.COVARIANCE:
            q_bar_matrix = np.cov(data_array)
        elif method == GarchMethod.MEAN:
            mean_returns = data_array.mean(axis=1)
            q_bar_matrix = np.outer(mean_returns, mean_returns)
        elif method == GarchMethod.EWMA:
            # Calculate exponentially weighted moving average covariance matrix
            # Assuming a decay factor (lambda) of 0.94, which is common for financial data
            decay_factor = 0.94
            q_bar_matrix = pd.DataFrame(data_array).ewm(alpha=1 - decay_factor).cov().to_numpy()[-rows:]

        covariance_matrix = [q_bar_matrix * (1 - alpha - beta)]
        correlation_matrix_list: list[Any] = []

        for t in range(1, rows + 1):
            z_vector = data_array[:, t - 1].reshape(-1, 1)
            z_matrix = z_vector @ z_vector.T

            covariance_matrix.append(
                q_bar_matrix * (1 - alpha - beta) + alpha * z_matrix + beta * covariance_matrix[t - 1]
            )
            diag_covariance = np.diag(np.diag(covariance_matrix[t]))
            non_negative_diagonal_covariance = np.where(diag_covariance <= 0, 1e-10, diag_covariance)

            correlation_matrix_list.append(
                np.linalg.inv(np.sqrt(non_negative_diagonal_covariance))
                @ covariance_matrix[t]
                @ np.linalg.inv(np.sqrt(non_negative_diagonal_covariance))
            )

        correlation_matrix = np.array(correlation_matrix_list)
        correlation_matrix_flat = correlation_matrix.reshape((rows, self.column_count**2))

        correlation_matrix_pd = pd.DataFrame(correlation_matrix_flat)
        correlation_matrix_pd.columns = [
            f"{self.names[i]}_{self.names[j]}" for i in range(self.column_count) for j in range(self.column_count)
        ]
        return correlation_matrix_pd.set_index(data_process.index)

    def calculate(self, params: list[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the epsilons and sigma for the GARCH model given the parameters.

        Args:
            params (list[Any]): A list of parameters for the GARCH model, including gamma, phi, omega, alpha, and beta.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames, epsilons and sigma, representing the residuals and conditional variances respectively.

        """
        rows = self.data.shape[0]
        columns = self.data.shape[1]

        gamma = params[:columns]
        phi = params[columns : columns * 2]
        omega = params[columns * 2 : columns * 3]
        alpha = params[columns * 3 : columns * 4]
        beta = params[columns * 4 :]

        epsilons = np.zeros(self.reversed_shape)
        sigma = np.zeros(self.reversed_shape)

        for i in range(columns):
            epsilons[(i, 0)] = self.data.iloc[(1, i)] - (gamma[i] + phi[i] * self.data.iloc[(0, i)])
            sigma[(i, 0)] = omega[i]

        for i in range(columns):
            for t in range(1, rows):
                sigma[(i, t)] = omega[i] + alpha[i] * epsilons[(i, t - 1)] ** 2 + beta[i] * sigma[(i, t - 1)]
                epsilons[(i, t)] = self.data.iloc[(t, i)] - (gamma[i] + phi[i] * self.data.iloc[(t - 1, i)])

        epsilons_df = pd.DataFrame(epsilons.T)
        epsilons_df.columns = self.names
        epsilons_df = epsilons_df.set_index(self.data.index)

        sigma_df = pd.DataFrame(sigma.T)
        sigma_df.columns = self.names
        sigma_df = sigma_df.set_index(self.data.index)

        return epsilons_df, sigma_df

    @staticmethod
    def callback(_: np.ndarray) -> None:
        """Increment the iteration count and print the current parameters during optimization."""
        Garch.iteration += 1
        logger.info(f"Iteration {Garch.iteration}")

    def minimize_correlation(self, sigma: pd.DataFrame, method: GarchMethod = GarchMethod.CORRELATION) -> pd.DataFrame:
        """Minimize the log-likelihood of the GARCH model with correlation to find the optimal alpha and beta parameters.

        Args:
            sigma (pd.DataFrame): The conditional variances for each time step and variable.
            method (GarchMethod): The method to use for calculating the correlation matrix. Options are "correlation", "covariance", "mean", and "ewma".

        Returns:
            pd.DataFrame: A DataFrame containing the optimized alpha and beta parameters for the GARCH model.

        Raises:
            ValueError: If the optimization fails.

        """
        params = [self.alpha_init, self.beta_init]
        bounds: list[tuple[Any, ...]] = [self.alpha_bounds_init, self.beta_bounds_init]
        constraints: list[dict[str, Any]] = [{"type": "ineq", "fun": lambda x: 1 - x[0] - x[1]}]

        # Drop the first row of the sigma and data because the first row of sigma is not calculated
        data_process = self.data.iloc[1:, :]
        sigma_process = sigma.iloc[1:, :]

        results = minimize(  # type: ignore[call-overload]
            Garch.calculate_correlation,
            params,
            args=(data_process, sigma_process, method),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            callback=Garch.callback,
            options={
                "disp": True,
                "ftol": 1e-9,  # Increased precision for function tolerance
                "eps": 1e-9,  # Step size used for numerical approximation of the jacobian
            },
        )

        logger.info(results.message)

        if not results.success:
            msg = "Optimization failed"
            raise ValueError(msg)

        optimized_params = pd.DataFrame(np.array([results.x]))

        param_names = ["alpha", "beta"]
        column_names = [f"{param}_{method}" for param in param_names]

        optimized_params.columns = column_names

        optimized_params.to_excel(OUTPUT_PATH / f"garch_params_{method}.xlsx")

        logger.info("Optimized parameters:")
        logger.info(optimized_params)

        return optimized_params

    def minimize(self) -> pd.DataFrame:
        """Minimize the log-likelihood of the GARCH model to find the optimal parameters.

        Returns:
            pd.DataFrame: A DataFrame containing the optimized parameters for the GARCH model.

        Raises:
            ValueError: If the optimization fails.

        """
        params = self.gamma + self.phi + self.omega + self.alpha + self.beta
        bounds: list[tuple[Any, ...]] = (
            self.gamma_bounds + self.phi_bounds + self.omega_bounds + self.alpha_bounds + self.beta_bounds
        )

        results = minimize(  # type: ignore[call-overload]
            Garch.calculate_individual,
            params,
            args=(self.data,),
            bounds=bounds,
            callback=Garch.callback,
            options={
                "disp": True,
                "ftol": 1e-9,  # Increased precision for function tolerance
                "eps": 1e-9,  # Step size used for numerical approximation of the jacobian
            },
        )

        logger.info(results.message)

        if not results.success:
            msg = "Optimization failed"
            raise ValueError(msg)

        optimized_params = pd.DataFrame(np.array([results.x]))

        param_names = ["gamma", "phi", "omega", "alpha", "beta"]
        column_names = [f"{param}_{name}" for param in param_names for name in self.names]

        optimized_params.columns = column_names
        optimized_params.to_excel(OUTPUT_PATH / "garch_params.xlsx")

        logger.info("Optimized parameters:")
        logger.info(optimized_params)

        return optimized_params
