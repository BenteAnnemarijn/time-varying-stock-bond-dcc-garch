from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

from thesis.config import PLOTS_PATH


def plot_df(df: pd.DataFrame, column: str | None, title: str, save_path: Path | None = None) -> None:
    """Plot a dataframe column over time."""
    plt.style.use("seaborn-v0_8-darkgrid")
    _, ax = plt.subplots(figsize=(12, 6))
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    if column is None:
        ax.plot(df.index, df, linestyle="-", linewidth=2, label=title)
    else:
        ax.plot(df.index, df[column], label=column, linestyle="-", linewidth=2)
    ax.legend(frameon=False, fontsize=12)

    plt.xlabel("Date", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.title(title, fontsize=16, pad=20)

    if save_path:
        save_path = save_path.with_suffix(".png")
        save_path = PLOTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
        input("Press Enter to continue...")
    plt.close()


def plot_break_points(
    df: pd.DataFrame, column: str, breakpoints: pd.DataFrame, title: str, save_path: Path | None = None
) -> None:
    """Plot a dataframe column over time with breakpoints."""
    plt.style.use("seaborn-v0_8-darkgrid")
    _, ax = plt.subplots(figsize=(12, 6))
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.plot(df.index, df[column], label=column, linestyle="-", linewidth=2)
    ax.legend(frameon=False, fontsize=12)

    first_breakpoint = True
    for bkpt in breakpoints["breakpoints"].to_numpy():
        if first_breakpoint:
            ax.axvline(x=df.index[bkpt - 1], color="steelblue", linestyle="--", linewidth=2, label="Breakpoint")
            first_breakpoint = False
        else:
            ax.axvline(x=df.index[bkpt - 1], color="steelblue", linestyle="--", linewidth=2)

    plt.xlabel("Date", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.title(title, fontsize=16, pad=20)

    if save_path:
        save_path = save_path.with_suffix(".png")
        save_path = PLOTS_PATH / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
        input("Press Enter to continue...")
    plt.close()
