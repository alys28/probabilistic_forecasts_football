# Given all the saved SHAP values, we will plot the importance of each feature over time individually
import numpy as np
import shap
import matplotlib.pyplot as plt

FEATURES = ["game_completed", "relative_strength", "score_difference", "type.id", "home_has_possession", "end.down", "end.yardsToEndzone", "end.distance", "field_position_shift", "home_timeouts_left", "away_timeouts_left"]

def open_shap_file(path):
    """
    Open SHAP file (numpy)
    """
    npz = np.load(path, allow_pickle=True)
    shap_output = shap.Explanation(
        values=npz["values"],
        base_values=npz["base_values"],
        data=npz["data"],
        feature_names=npz["feature_names"].tolist(),
    )
    return shap_output

class SHAP_over_time:
    def __init__(self):
        """
        self.__lines: Dictionary with key as the feature name, and value as a list of SHAP values, one for each timestep
        """
        self.__lines = {}

    def plot(self, plot_title="SHAP over time", save_path=None, show=True):
        """
        Plot all lines with x-axis from 0 to 1. If series have different lengths,
        they are interpolated onto a common grid for visual comparison.

        Args:
            plot_title: Title for the plot.
            save_path: Optional path to save the figure (e.g., 'shap_over_time.png').
            show: Whether to display the plot (True) or just build it (False).
        """
        if not self.__lines:
            raise ValueError("No lines to plot. Use add_line(feature_name, values) first.")

        # Choose a common grid size (max length among series)
        max_len = max(len(v) for v in self.__lines.values())
        x_common = np.linspace(0.0, 1.0, max_len)

        plt.figure(figsize=(9, 5))
        for name, y in self.__lines.items():
            # Original grid for this series
            x_orig = np.linspace(0.0, 1.0, len(y))
            # Interpolate to common grid if needed
            if len(y) != max_len:
                y_plot = np.interp(x_common, x_orig, y)
            else:
                y_plot = y
            plt.plot(x_common, y_plot, label=name)

        plt.title(plot_title)
        plt.xlabel("Relative time (0 → 1)")
        plt.ylabel("SHAP value")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Feature", loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return x_common

    def add_line(self, feature_name, values):
        assert feature_name not in self.__lines, "Feature already exists in the set of lines."
        self.__lines[feature_name] = values

