# Time-Varying Stock-Bond DCC-GARCH Analysis

A comprehensive research project analyzing the dynamic correlations between stock and bond markets using time-varying Dynamic Conditional Correlation (DCC) and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models.

## Overview

This project implements a complete pipeline for analyzing the relationship between financial asset classes (stocks and bonds) while accounting for:

- **Time-varying correlations**: Rolling window correlation analysis
- **Volatility clustering**: GARCH model optimization
- **Structural breaks**: Kernel-based change point detection
- **Market regimes**: Analysis of Fed funds rate and inflation dynamics

The analysis combines multiple statistical tests and visualization techniques to provide insights into market behavior and asset class relationships over time.

## Features

- **Data Processing**: Load and clean multiple financial datasets (stocks, bonds, Fed funds, inflation)
- **Statistical Analysis**:
  - Stationarity tests (ADF, KPSS)
  - Jarque-Bera normality tests
  - Rolling window correlation
- **GARCH Modeling**: DCC-GARCH model optimization for capturing time-varying correlations and volatility
- **Breakpoint Detection**: Kernel-based methods for identifying structural breaks in market behavior
- **Visualization**: Comprehensive plotting of correlations, volatility, breakpoints, and market regimes
- **Flexible Pipeline**: Modular command-line interface for selective execution of analysis components

## Project Structure

```
.
├── main.py                      # Main entry point with CLI arguments
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Locked dependency versions
├── thesis/
│   ├── config.py               # Configuration (paths, environment variables)
│   ├── analysis/
│   │   ├── breakpoints.py      # Structural break detection
│   │   └── correlation.py      # Rolling window correlation calculation
│   ├── lib/
│   │   ├── data_processing.py  # Base data loader class
│   │   ├── visualization.py    # Plotting utilities
│   │   ├── data/
│   │   │   ├── bond.py         # Bond data loading
│   │   │   ├── stock.py        # Stock data loading
│   │   │   ├── fed_fund.py     # Fed funds rate loading
│   │   │   └── inflation.py    # Inflation data loading
│   │   └── tests/
│   │       ├── stationary.py   # Stationarity tests
│   │       └── bera.py         # Jarque-Bera tests
│   ├── models/
│   │   └── garch.py            # DCC-GARCH model implementation
│   └── utils/
│       ├── helpers.py          # Utility functions
│       └── logger.py           # Logging configuration
├── data/
│   ├── input/                  # Raw input data
│   └── output/                 # Processed results and plots
└── logs/                       # Application logs
```

## Installation

### Requirements

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd time-varying-stock-bond-dcc-garch
```

1. Install dependencies with uv:

```bash
uv sync
```

This will create a virtual environment and install all dependencies.

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
uv run main.py
```

### Command-Line Options

```
-p, --process-data          Process raw data and save cleaned versions (default: True)
-P, --skip-process-data     Skip data processing, load from cache

-t, --select-timeframe      Select specific timeframe (2000-01-01 to 2021-01-01)

-c, --create-plots          Create visualization plots (default: True)
-C, --skip-plots            Skip plot creation

-g, --garch-model           Run GARCH model optimization (default: True)
-G, --skip-garch            Skip GARCH model, load from cache

-s, --test-stationary       Run stationarity tests on data

-j, --test-jarque-bera      Run Jarque-Bera tests on data

-d, --determine-correlation Calculate rolling window and correlation (default: True)
-D, --skip-correlation      Skip correlation analysis, load from cache

-b, --calculate-breaks      Calculate breakpoints using kernel methods (default: True)
-B, --skip-breaks           Skip breakpoint calculation

--start-date TEXT           Start date for timeframe (YYYY-MM-DD, default: 2000-01-01)
--end-date TEXT             End date for timeframe (YYYY-MM-DD, default: 2021-01-01)
```

### Example Workflows

**Process data only:**

```bash
uv run main.py --skip-plots --skip-garch --skip-correlation --skip-breaks
```

**Run analysis with custom timeframe:**

```bash
uv run main.py --select-timeframe --start-date 2010-01-01 --end-date 2020-12-31
```

**Skip data processing and use cache:**

```bash
uv run main.py --skip-process-data
```

**Run all statistical tests:**

```bash
uv run main.py --test-stationary --test-jarque-bera
```

## Data Sources

The project loads financial data from:

- **Stock Index**: ESIndex (S&P 500 equivalent)
- **Bond Data**: US Government bonds
- **Fed Funds Rate**: Federal Reserve fund rates
- **Inflation**: CPI/inflation indicators

Raw data should be placed in `data/input/` directory.

## Output

Results are saved to `data/output/`:

- **bond/**: Bond-specific analysis results
- **stock/**: Stock-specific analysis results
- **bond_vs_stock/**: Correlation and joint analysis
  - `rolling_window_correlation/`: Time-series correlation data
  - `break_points/`: Structural break locations
- **plots/**: Visualization outputs
- **fed_fund/**: Federal funds rate analysis
- **inflation/**: Inflation analysis

## Key Components

### Data Loader

Base class (`DataLoader`) for loading and processing financial time series with data validation and NaN handling.

### GARCH Model

Implements Dynamic Conditional Correlation (DCC) GARCH model for:

- Estimating time-varying conditional correlations
- Modeling volatility clustering
- Parameter optimization via maximum likelihood estimation

### Breakpoint Detection

Uses kernel-based methods (ruptures library) to identify:

- Structural breaks in correlations
- Regime changes in market behavior
- Change points in volatility patterns

### Statistical Tests

- **Stationarity**: ADF and KPSS tests
- **Normality**: Jarque-Bera test
- Log results and summary statistics

## Configuration

Edit `thesis/config.py` to customize:

- Data input path (via `DATA_PATH` environment variable)
- Output path (via `OUTPUT_PATH` environment variable)
- Plots path (via `PLOTS_PATH` environment variable)

Or create a `.env` file in the project root:

```
DATA_PATH=data/input
OUTPUT_PATH=data/output
PLOTS_PATH=data/output/plots
```

## Development

### Code Quality Tools

The project uses:

- **mypy**: Static type checking
- **ruff**: Linting and formatting

Run checks with uv:

```bash
uv run mypy thesis/
uv run ruff check thesis/
```

### Logging

The application uses color-coded logging. Logs are output to console and saved to `logs/` directory.

## Dependencies

Main dependencies:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and optimization
- **statsmodels**: Statistical tests and models
- **matplotlib**: Data visualization
- **ruptures**: Change point detection
- **python-dotenv**: Environment variable management

See `pyproject.toml` for complete list with versions. Dependencies are locked in `uv.lock` for reproducible builds.

## License

Master's Thesis Project

## Contact

For questions or issues, please refer to the thesis documentation.
