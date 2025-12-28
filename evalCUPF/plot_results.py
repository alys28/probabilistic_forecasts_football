import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def squared_error(y, p):
    return (y - p) ** 2

def calc_L_s2(df, pA="phat_1", pB="phat_2", Y="Y", grid="grid", L_func=squared_error):
    """
    Calculate pointwise mean loss difference and variance estimate
    """
    df = df.copy()

    # Compute si term
    si = (L_func(1, df[pA]) - L_func(0, df[pA])) - (L_func(1, df[pB]) - L_func(0, df[pB]))
    df["si"] = si

    # Group by grid / timestep
    grouped = df.groupby(grid).apply(
        lambda g: pd.Series({
            "L": np.mean(L_func(g[Y], g[pA]) - L_func(g[Y], g[pB])),
            "sigma2": np.mean(g["si"] ** 2) / 4,
            "n": len(g)
        })
    ).reset_index()

    return grouped


def plot_pcb(df, grid="grid", L="L", var="sigma2", phat_A="phat_A", phat_B="phat_B", pad=None):
    """
    Plotter for point-wise confidence bandW
    Args:
        df (pd.DataFrame): DataFrame containing the data
        grid (str): column name for x-axis
        L (str): column name for main line
        var (str): column name for variance
        phat_A (str): label for lower text annotation
        phat_B (str): label for upper text annotation
        pad (float, optional): padding for text labels. Defaults to 1% of y-range.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: the plot
    """
    # z-values for 95% confidence interval
    z_hi = norm.ppf(0.975)
    z_lo = norm.ppf(0.025)

    n = len(df)
    se = np.sqrt(df[var]) / np.sqrt(n)
    ymax = df[L] + z_hi * se
    ymin = df[L] + z_lo * se

    # Compute label positions
    y_top = ymax.max(skipna=True)
    y_bot = ymin.min(skipna=True)
    y_range = y_top - y_bot
    if pad is None:
        pad = 0.01 * y_range  # 1% of y-range

    # Determine x position for text labels
    x_vals = df[grid]
    try:
        x_pos = x_vals.max()
    except:
        x_pos = x_vals.iloc[-1]

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=grid, y=L, color="black")
    plt.fill_between(df[grid], ymin, ymax, color='red', alpha=0.2)
    plt.axhline(0, color='blue', linewidth=1.25)

    # Annotate labels
    plt.text(x_pos, y_bot - pad, f"{phat_A} favoured", ha='right', va='top', fontsize=12, color="black")
    plt.text(x_pos, y_top + pad, f"{phat_B} favoured", ha='right', va='bottom', fontsize=12, color="black")

    plt.xlabel(grid)
    plt.ylabel(L)
    plt.tight_layout()
    plt.show()
