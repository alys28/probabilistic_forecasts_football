# Given all the saved SHAP values, we will plot the importance of each feature over time individually
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

FEATURES = ["game_completed", "relative_strength", "score_difference", "type.id", "home_has_possession", "end.down", "end.yardsToEndzone", "end.distance", "field_position_shift", "home_timeouts_left", "away_timeouts_left"]

def load_SHAP_output(path):
    npz = np.load(path, allow_pickle=True)

    shap_outputs = {}

    # Extract unique timesteps from the keys
    timesteps = sorted(set(
        key.split("_", 1)[1]
        for key in npz.keys()
    ))

    for t in timesteps:
        shap_outputs[float(t)] = {
            "values": npz[f"values_{t}"],
            "base_values": npz[f"base_values_{t}"],
            "data": npz[f"data_{t}"],
            "feature_names": npz[f"feature_names_{t}"].tolist(),
        }

    return shap_outputs

class SHAP_over_time:
    def __init__(self):
        """
        self.__lines: Dictionary with key as the feature name, and value as a list of list of SHAP values, one for each model, each entry for a given corresponds to the SHAP value of a given model
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
        result = self.normalize_lines()
        # Choose a common grid size (max length among series)
        max_len = max(len(v) for v in result.values())
        x_common = np.linspace(0.0, 1.0, max_len)

        plt.figure(figsize=(9, 5))
        for name, y in result.items():
            x_orig = np.linspace(0.0, 1.0, len(y))
            if len(y) != max_len:
                y_plot = np.interp(x_common, x_orig, y)
            else:
                y_plot = y
            plt.plot(x_common, y_plot, label=name)

        plt.title(plot_title)
        plt.xlabel("Time")
        plt.ylabel("SHAP value")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Feature", loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return x_common

    def normalize_lines(self):
        """
        Return a dictionary containing one set of data points per feature, which have been scaled timestep wise between 0 and 1
        """
        idx = {}
        i = 0
        means = []
        for key, lists in self.__lines:
            lists = np.array(lists)
            col_means = np.mean(lists, axis=0)
            means.append(col_means)
            idx[key] = i
            i += 1
        means = np.array(means)
        means = means.transpose()
        normalized = means / means.sum(axis=1, keepdims=True)
        normalized = normalized.transpose
        result = {}
        for key, i in idx:
            result[key] = list(normalized[i])
        return result

    def __add_line(self, feature_name, values):
        if feature_name in self.__lines:
            self.__lines[feature_name].append(values)
        else:
            self.__lines[feature_name] = [values]

    def process_np_file(file: str):


if __name__ == "__main__":
    shap_dir = ""
    shap_over_time = SHAP_over_time()
    for file in os.listdir(shap_dir):
        file_dir = os.path.join(shap_dir, file)