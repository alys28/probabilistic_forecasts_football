import numpy as np


class KernelKNN:
    def __init__(self, kernel_function, k=5):
        self.kernel_function = kernel_function
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> int:
        similarities = self.kernel_function.compute(self.X_train, x)
        k_nearest_indices = np.argsort(similarities)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        # Weighted average of the k nearest labels
        weights = similarities[k_nearest_indices]
        return np.sum(k_nearest_labels * weights) / np.sum(weights)