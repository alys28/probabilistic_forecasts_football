import numpy as np

from models.Model import Model


class NFLHeuristicBucketerModel(Model):
    """
    Heuristic bucketer that mirrors evalCUPF/NFL_example/nfl_heuristic_bucketer.py.
    It returns per-sample bucket win probabilities (mean label per bucket).
    """

    REQUIRED_FEATURES = {
        "score_difference",
        "relative_strength",
        "end.yardsToEndzone",
        "home_has_possession",
    }

    def __init__(self, features, n_buckets=5, use_calibration=False):
        super().__init__(use_calibration=use_calibration, optimize_hyperparams=False)
        self.features = list(features)
        self.feature_map = {feat: idx for idx, feat in enumerate(self.features)}
        self.n_features = len(self.features)
        self.n_buckets = int(n_buckets)
        if self.n_buckets != 5:
            raise ValueError("NFLHeuristicBucketerModel uses exactly 5 buckets.")

        missing = [f for f in self.REQUIRED_FEATURES if f not in self.feature_map]
        if missing:
            raise ValueError(
                f"Missing required feature(s) for heuristic bucketer: {missing}"
            )

        self.start = 0.0
        self.end = 1.0
        self.buckets = {}
        self.v = {}
        self.quantiles = None

    def _define_search_space(self, trial):
        return {}

    def _train_model(self, x_train, y_train, x_val, y_val, params):
        return 0.0

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

    def _get_feature(self, row, name):
        if name in self.feature_map:
            return row[self.feature_map[name]]
        return None

    def _compute_pressure_scores(self, data):
        """Compute heuristic pressure scores for each row."""
        n = data.shape[0]
        scores = np.zeros(n)
        if self.end <= 0.25:
            phase = "early"
        elif self.end <= 0.55:
            phase = "mid"
        elif self.end <= 0.85:
            phase = "late"
        else:
            phase = "critical"

        for i, row in enumerate(data):
            score_diff = self._get_feature(row, "score_difference")
            rel_strength = self._get_feature(row, "relative_strength")
            yards_to_endzone = self._get_feature(row, "end.yardsToEndzone")
            home_possession = self._get_feature(row, "home_has_possession")

            pressure = -score_diff

            if home_possession:
                field = max(0, (40 - yards_to_endzone) / 40)
            else:
                field = max(0, (60 - yards_to_endzone) / 30)

            if phase == "early":
                pressure += 0.6 * field
                pressure += -0.7 * rel_strength
                pressure += 0.2 * (not home_possession)
            elif phase == "mid":
                pressure += 0.9 * field
                pressure += -0.4 * rel_strength
                pressure += 0.5 * (not home_possession)
            elif phase == "late":
                pressure += 1.3 * field
                pressure += 0.8 * (not home_possession)
            else:
                pressure *= 1.6
                pressure += 1.5 * field
                pressure += 1.2 * (not home_possession)

            scores[i] = pressure

        return scores

    @staticmethod
    def _assign_from_scores(scores, qs):
        n = scores.shape[0]
        assignments = np.zeros(n, dtype=int)
        for i, s in enumerate(scores):
            if s <= qs[0]:
                assignments[i] = 0
            elif s <= qs[1]:
                assignments[i] = 1
            elif s <= qs[2]:
                assignments[i] = 2
            elif s <= qs[3]:
                assignments[i] = 3
            else:
                assignments[i] = 4

        return assignments

    def _assign_buckets(self, data):
        """
        Assign buckets using fixed training quantiles when available.
        """
        scores = self._compute_pressure_scores(data)
        if self.quantiles is None:
            qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
        else:
            qs = self.quantiles
        return self._assign_from_scores(scores, qs)

    def _fit_strategy(self, data, labels):
        if data.size == 0:
            self.buckets = {}
            self.v = {}
            self.quantiles = None
            return

        scores = self._compute_pressure_scores(data)
        self.quantiles = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
        assignments = self._assign_from_scores(scores, self.quantiles)
        self.v = {}
        for bucket_id in range(self.n_buckets):
            bucket_name = f"bucket_{bucket_id}"
            mask = assignments == bucket_id
            n_j_t = int(np.sum(mask))
            if n_j_t > 0:
                y_mean_t = float(np.mean(labels[mask]))
                self.v[bucket_name] = y_mean_t
            else:
                self.v[bucket_name] = 0.0

        self.buckets = {f"bucket_{i}": i for i in range(self.n_buckets)}

    def fit(self, X, y, val_X=None, val_y=None, timestep=None, start=None, end=None):
        labels = np.asarray(y, dtype=np.float64).reshape(-1)
        if np.isnan(labels).any():
            raise ValueError("NaN values found in target variable")

        features = self._extract_current_features(X)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatched rows between X and y: {features.shape[0]} vs {labels.shape[0]}"
            )
        if features.shape[0] <= self.n_buckets:
            raise ValueError(
                f"Need more data for the given bucket. Got {features.shape[0]} data and {self.n_buckets} buckets"
            )

        ref = float(timestep) if timestep is not None else 0.0
        self.start = float(start) if start is not None else ref
        self.end = float(end) if end is not None else ref

        self._fit_strategy(features, labels)
        self.model = self

        if (
            self.use_calibration
            and val_X is not None
            and val_y is not None
            and len(val_y) > 0
        ):
            val_scores = np.clip(self._predict_values(val_X), 0.0, 1.0)
            self.fit_calibrator(val_scores, np.asarray(val_y, dtype=np.float64).reshape(-1))

    def _score(self, X):
        if len(self.buckets) == 0:
            X_arr = self._extract_current_features(X)
            return np.zeros((X_arr.shape[0], 0), dtype=np.float64)

        X_arr = self._extract_current_features(X)
        scores = np.zeros((X_arr.shape[0], self.n_buckets), dtype=np.float64)
        assignments = self._assign_buckets(X_arr)
        for i, bucket_id in enumerate(assignments):
            scores[i, bucket_id] = 1.0
        return scores

    def _predict_values(self, X):
        scores = self._score(X)
        if scores.shape[1] == 0:
            return np.zeros(scores.shape[0], dtype=np.float64)

        bucket_names = list(self.buckets.keys())
        best_bucket_indices = np.argmax(scores, axis=1)
        best_buckets = [bucket_names[i] for i in best_bucket_indices]
        return np.array([self.v[b] for b in best_buckets], dtype=np.float64)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        probs = self._predict_values(X)
        probs = np.clip(probs, 0.0, 1.0)
        if self.use_calibration and self.calibrator is not None:
            probs = self.apply_calibration(probs)
        return np.column_stack([1 - probs, probs])


def setup_nfl_heuristic_bucket_models(
    training_data,
    test_data=None,
    features=None,
    n_buckets=5,
    use_calibration=False,
):
    """
    Train one NFLHeuristicBucketerModel per timestep key in training_data.
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

        if X.shape[0] <= n_buckets:
            print(
                f"Skipping timestep {timestep}: need more than {n_buckets} samples, got {X.shape[0]}"
            )
            continue

        model = NFLHeuristicBucketerModel(
            features=features,
            n_buckets=n_buckets,
            use_calibration=use_calibration,
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
