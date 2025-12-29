import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def squared_error(y, p):
    return (y - p) ** 2

def calc_L_s2(
    df,
    C1,
    C2,
    pA="phat_1",
    pB="phat_2",
    Y="Y",
    grid="grid",
    L_func=squared_error
):
    """
    Calculate pointwise mean loss difference and variance
    using precomputed covariance matrices C1 and C2
    """

    df = df.copy()

    # Pointwise mean loss difference
    L_df = (
        df.groupby(grid)
          .apply(lambda g: np.mean(L_func(g[Y], g[pA]) - L_func(g[Y], g[pB])))
          .rename("L")
          .reset_index()
    )

    # Sample size per grid
    n_df = (
        df.groupby(grid)
          .size()
          .rename("n")
          .reset_index()
    )

    # Extract diagonal of covariance matrices
    if isinstance(C1, pd.DataFrame) and isinstance(C2, pd.DataFrame):
        sigma2_df = (
            pd.DataFrame({
                grid: C1.index,
                "sigma2_C1": np.diag(C1.values),
                "sigma2_C2": np.diag(C2.values)
            })
        )
    else:
        sigma2_df = pd.DataFrame({
            grid: np.unique(df[grid]),
            "sigma2_C1": np.diag(C1),
            "sigma2_C2": np.diag(C2)
        })

    # Merge everything
    out = (
        L_df
        .merge(sigma2_df, on=grid)
        .merge(n_df, on=grid)
    )

    return out


def plot_pcb(df, grid="grid", L="L", var_C1="sigma2_C1", var_C2="sigma2_C2", phat_A="phat_A", phat_B="phat_B", pad=None):
    """
    Plotter for point-wise confidence band using variances from C1 and C2
    """
    # z-values for 95% confidence interval
    z_hi = norm.ppf(0.975)
    z_lo = norm.ppf(0.025)

    # Check if required columns exist
    if not all(col in df.columns for col in [grid, L, var_C1, var_C2]):
        raise ValueError(f"DataFrame must contain columns: {grid}, {L}, {var_C1}, {var_C2}")

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty.")

    # Calculate standard error and confidence interval bounds for C1 and C2
    se_C1 = np.sqrt(df[var_C1]) / np.sqrt(n)
    se_C2 = np.sqrt(df[var_C2]) / np.sqrt(n)

    if se_C1.isnull().any() or se_C2.isnull().any():
        raise ValueError("Standard error contains NaN values. Check the variance columns.")

    ymax_C1 = df[L] + z_hi * se_C1
    ymin_C1 = df[L] + z_lo * se_C1
    ymax_C2 = df[L] + z_hi * se_C2
    ymin_C2 = df[L] + z_lo * se_C2

    # Compute label positions
    y_top = max(ymax_C1.max(skipna=True), ymax_C2.max(skipna=True))
    y_bot = min(ymin_C1.min(skipna=True), ymin_C2.min(skipna=True))
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
    sns.lineplot(data=df, x=grid, y=L, color="black", label="Mean Loss Difference")
    plt.fill_between(df[grid], ymin_C1, ymax_C1, color='red', alpha=0.2, label="95% CI (C1)")
    plt.fill_between(df[grid], ymin_C2, ymax_C2, color='blue', alpha=0.2, label="95% CI (C2)")
    plt.axhline(0, color='black', linewidth=1.25, linestyle="--")

    # Annotate labels
    plt.text(x_pos, y_bot - pad, f"{phat_A} favoured", ha='right', va='top', fontsize=12, color="black")
    plt.text(x_pos, y_top + pad, f"{phat_B} favoured", ha='right', va='bottom', fontsize=12, color="black")

    plt.xlabel(grid)
    plt.ylabel(L)
    plt.legend()
    plt.tight_layout()
    plt.show()