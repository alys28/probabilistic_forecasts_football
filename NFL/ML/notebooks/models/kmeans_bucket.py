import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from models.Model import Model


class KMeansBucketerModel(Model):
    """
    KMeans bucketer that mirrors evalCUPF/NFL_example/nfl_bucketer.py.
    It returns per-sample bucket positive-label means as class-1 output.
    """
    MAX_CLUSTERS = 15

    def __init__(
        self,
        features,
        n_buckets=3,
        random_state=42,
        use_calibration=False,
        optimize_hyperparams=False,
        n_trials=50,
        min_n_buckets=2,
        max_n_buckets=15,
    ):
        super().__init__(
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
        )
        self.features = list(features)
        self.n_features = len(self.features)
        self.n_buckets = int(n_buckets)
        self.random_state = random_state
        self.min_n_buckets = int(min_n_buckets)
        requested_max = self.MAX_CLUSTERS if max_n_buckets is None else int(max_n_buckets)
        self.max_n_buckets = min(requested_max, self.MAX_CLUSTERS)
        self._hpo_min_buckets = max(2, self.min_n_buckets)
        self._hpo_max_buckets = max(self._hpo_min_buckets, self.n_buckets)

        self.start = 0.0
        self.end = 1.0

        self.scaler = None
        self.kmeans = None
        self.buckets = {}
        self.v = {}
        self.bucket_positive_mean = {}

    def _define_search_space(self, trial):
        return {
            "n_buckets": trial.suggest_int(
                "n_buckets",
                self._hpo_min_buckets,
                self._hpo_max_buckets,
            )
        }

    def _fixed_params(self):
        return {"random_state": self.random_state}

    def _compute_bucket_state(self, data, labels, n_buckets):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_buckets, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_scaled)

        buckets = {f"bucket_{i}": kmeans.cluster_centers_[i] for i in range(n_buckets)}
        bucket_positive_mean = {}
        for j in range(n_buckets):
            mask = cluster_labels == j
            y_mean_t = float(np.mean(labels[mask])) if np.any(mask) else 0.0
            bucket_positive_mean[f"bucket_{j}"] = y_mean_t
        return scaler, kmeans, buckets, bucket_positive_mean

    @staticmethod
    def _count_unique_rows(X):
        if X.shape[0] == 0:
            return 0
        return int(np.unique(X, axis=0).shape[0])

    def _predict_values_with_state(self, X, scaler, buckets, bucket_positive_mean):
        if len(buckets) == 0:
            X_arr = self._extract_current_features(X)
            return np.zeros(X_arr.shape[0], dtype=np.float64)

        X_arr = self._extract_current_features(X)
        X_scaled = scaler.transform(X_arr)
        bucket_names = list(buckets.keys())
        centroids = np.array([buckets[b] for b in bucket_names], dtype=np.float64)
        scores = cosine_similarity(X_scaled, centroids)
        best_bucket_indices = np.argmax(scores, axis=1)
        best_buckets = [bucket_names[i] for i in best_bucket_indices]
        return np.array([bucket_positive_mean[b] for b in best_buckets], dtype=np.float64)

    def _train_model(self, x_train, y_train, x_val, y_val, params):
        n_buckets = int(params["n_buckets"])
        unique_rows = self._count_unique_rows(x_train)
        if x_train.shape[0] <= n_buckets or unique_rows < n_buckets:
            return float("inf")
        scaler, _, buckets, bucket_positive_mean = self._compute_bucket_state(
            x_train, y_train, n_buckets
        )
        y_val_pred = np.clip(
            self._predict_values_with_state(x_val, scaler, buckets, bucket_positive_mean),
            1e-15,
            1 - 1e-15,
        )
        y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
        val_loss = -np.mean(
            y_val * np.log(y_val_pred) + (1 - y_val) * np.log(1 - y_val_pred)
        )
        return float(val_loss)

    def _extract_current_features(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            if self.n_features > 0 and X.size % self.n_features == 0:
                X = X.reshape(1, -1, self.n_features)[:, -1, :]
            else:
                X = X.reshape(1, -1)
        elif X.ndim == 2:
            if self.n_features > 0 and X.shape[1] != self.n_features:
                if X.shape[1] % self.n_features == 0:
                    X = X.reshape(X.shape[0], -1, self.n_features)[:, -1, :]
                else:
                    raise ValueError(
                        f"Cannot map input shape {X.shape} to feature width {self.n_features}."
                    )
        elif X.ndim == 3:
            X = X[:, -1, :]
        else:
            raise ValueError(f"Unsupported input rank: {X.ndim}")

        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}.")
        return np.nan_to_num(X, nan=0.0)

    def _fit_strategy(self, data, labels):
        if data.size == 0:
            self.buckets = {}
            self.v = {}
            self.bucket_positive_mean = {}
            return

        self.scaler, self.kmeans, self.buckets, self.bucket_positive_mean = (
            self._compute_bucket_state(data, labels, self.n_buckets)
        )

        # Keep v available for compatibility/debugging.
        self.v = dict(self.bucket_positive_mean)

    def fit(self, X, y, val_X=None, val_y=None, timestep=None, start=None, end=None):
        labels = np.asarray(y, dtype=np.float64).reshape(-1)
        if np.isnan(labels).any():
            raise ValueError("NaN values found in target variable")

        features = self._extract_current_features(X)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatched rows between X and y: {features.shape[0]} vs {labels.shape[0]}"
            )
        if features.shape[0] <= 2:
            raise ValueError(
                f"Need more data for KMeans bucketing. Got only {features.shape[0]} rows."
            )

        ref = float(timestep) if timestep is not None else 0.0
        self.start = float(start) if start is not None else ref
        self.end = float(end) if end is not None else ref

        if val_X is None and val_y is None:
            X_train, X_val, y_train, y_val = self.split_data(
                features, labels, test_size=0.15, random_state=42, stratify=False
            )
        else:
            X_train, y_train = features, labels
            X_val = self._extract_current_features(val_X)
            y_val = np.asarray(val_y, dtype=np.float64).reshape(-1)
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"Mismatched rows between val_X and val_y: {X_val.shape[0]} vs {y_val.shape[0]}"
                )

        max_feasible = X_train.shape[0] - 1
        if max_feasible < 2:
            raise ValueError(
                f"Need more training rows to fit KMeans buckets. Got {X_train.shape[0]} rows."
            )
        max_distinct = self._count_unique_rows(X_train)
        if max_distinct < 2:
            raise ValueError(
                "Need at least 2 distinct rows in training data to fit KMeans buckets."
            )
        feasible_upper = min(max_feasible, max_distinct)

        lower = max(2, self.min_n_buckets)
        upper = (
            feasible_upper
            if self.max_n_buckets is None
            else min(self.max_n_buckets, feasible_upper)
        )
        if lower > upper:
            lower = upper
        self._hpo_min_buckets = int(lower)
        self._hpo_max_buckets = int(upper)

        if self.optimize_hyperparams and X_val.shape[0] > 0:
            best_params = self.optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=self.n_trials,
                direction="minimize",
            )
            self.n_buckets = int(best_params["n_buckets"])
        else:
            self.n_buckets = int(np.clip(self.n_buckets, lower, upper))

        self._fit_strategy(X_train, y_train)
        self.model = self

        if self.use_calibration and X_val.shape[0] > 0:
            val_scores = np.clip(self._predict_values(X_val), 0.0, 1.0)
            self.fit_calibrator(val_scores, y_val)

    def _predict_values(self, X):
        if len(self.buckets) == 0:
            X_arr = self._extract_current_features(X)
            return np.zeros(X_arr.shape[0], dtype=np.float64)

        X_arr = self._extract_current_features(X)
        X_scaled = self.scaler.transform(X_arr)

        bucket_names = list(self.buckets.keys())
        centroids = np.array([self.buckets[b] for b in bucket_names], dtype=np.float64)
        scores = cosine_similarity(X_scaled, centroids)
        best_bucket_indices = np.argmax(scores, axis=1)
        best_buckets = [bucket_names[i] for i in best_bucket_indices]
        return np.array([self.bucket_positive_mean[b] for b in best_buckets], dtype=np.float64)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        probs = self._predict_values(X)
        if self.use_calibration and self.calibrator is not None:
            probs = self.apply_calibration(np.clip(probs, 0.0, 1.0))
        return np.column_stack([1 - probs, probs])


def setup_kmeans_bucket_models(
    training_data,
    test_data=None,
    features=None,
    n_buckets=3,
    random_state=42,
    use_calibration=False,
    optimize_hyperparams=False,
    n_trials=50,
    min_n_buckets=2,
    max_n_buckets=15,
):
    """
    Train one KMeansBucketerModel per timestep key in training_data.
    """
    if features is None:
        features = []

    # Align validation routing with inference/evaluation:
    # choose validation rows by their assigned "model" key when available.
    test_rows_by_model = {}
    if test_data is not None:
        for _, entries in test_data.items():
            for row in entries:
                model_key = row.get("model")
                if model_key is None:
                    continue
                model_key = round(float(model_key), 3)
                if model_key not in test_rows_by_model:
                    test_rows_by_model[model_key] = []
                test_rows_by_model[model_key].append(row)

    models = {}
    for timestep in training_data:
        rows = training_data[timestep]
        if len(rows) == 0:
            continue

        X = np.asarray([np.asarray(row["rows"], dtype=np.float32) for row in rows], dtype=np.float32)
        y = np.asarray([row["label"] for row in rows], dtype=np.float64)

        X_test = None
        y_test = None
        timestep_key = round(float(timestep), 3)
        model_test_rows = test_rows_by_model.get(timestep_key, [])
        if len(model_test_rows) > 0:
            X_test = np.asarray(
                [np.asarray(row["rows"], dtype=np.float32) for row in model_test_rows],
                dtype=np.float32,
            )
            y_test = np.asarray([row["label"] for row in model_test_rows], dtype=np.float64)
        elif test_data is not None and timestep in test_data and len(test_data[timestep]) > 0:
            # Fallback for datasets without "model" assignment.
            X_test = np.asarray(
                [np.asarray(row["rows"], dtype=np.float32) for row in test_data[timestep]],
                dtype=np.float32,
            )
            y_test = np.asarray([row["label"] for row in test_data[timestep]], dtype=np.float64)

        min_required_buckets = (
            max(2, int(min_n_buckets))
            if optimize_hyperparams
            else max(2, int(n_buckets))
        )
        if X.shape[0] <= min_required_buckets:
            print(
                f"Skipping timestep {timestep}: need more than {min_required_buckets} samples, got {X.shape[0]}"
            )
            continue

        model = KMeansBucketerModel(
            features=features,
            n_buckets=n_buckets,
            random_state=random_state,
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
            min_n_buckets=min_n_buckets,
            max_n_buckets=max_n_buckets,
        )

        print(f"Timestep {float(timestep):.3f}", end=" ")
        model.fit(X, y, val_X=X_test, val_y=y_test, timestep=float(timestep))

        train_probs = np.clip(model.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
        train_loss = -np.mean(y * np.log(train_probs) + (1 - y) * np.log(1 - train_probs))
        train_acc = float(np.mean((train_probs > 0.5).astype(int) == y))

        if X_test is not None and y_test is not None and len(y_test) > 0:
            test_probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-15, 1 - 1e-15)
            test_loss = -np.mean(y_test * np.log(test_probs) + (1 - y_test) * np.log(1 - test_probs))
            test_acc = float(np.mean((test_probs > 0.5).astype(int) == y_test))
            print(
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
            )
        else:
            print(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        models[timestep] = model

    return models
