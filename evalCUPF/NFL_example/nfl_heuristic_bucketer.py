from evalCUPF.risk_buckets import Bucketer
import numpy as np

class NFLHeuristicBucketer(Bucketer):
    def __init__(self, features, data, labels, start, end, n_buckets=5):
        """
        Bucket data using NFL-specific heuristics based on game situation.
        Creates exactly n_buckets (default 5) based on combinations of:
        - Score margin and relative strength
        - Field position
        - Down and distance situation
        """
        self.n_buckets = n_buckets
        assert len(data) > n_buckets, f"Need more data for the given bucket. Got {len(data)} data and {n_buckets} buckets"
        super().__init__(features, data, labels=labels, start=start, end=end)
    
    def _preprocess_strategy(self, data, labels):
        """
        Define exactly 5 buckets based on NFL game situation heuristics.
        Each bucket represents a distinct game scenario cluster.
        """
        if data.size == 0:
            self.buckets = {}
            self.v = {}
            return
        
        # Extract feature indices
        self.feature_map = {feat: idx for idx, feat in enumerate(self.features)}
        
        # Define bucket assignment function
        self.bucket_assignments = self._assign_buckets(data)
        
        # Create bucket rules and compute statistics
        for bucket_id in range(self.n_buckets):
            bucket_name = f"bucket_{bucket_id}"
            mask = self.bucket_assignments == bucket_id
            n_j_t = np.sum(mask)
            
            if n_j_t > 0:
                y_mean_t = np.mean(labels[mask])
                self.add_to_v(bucket_name, y_mean_t, n_j_t)
            else:
                self.add_to_v(bucket_name, 0.0, 0)
        
        # Store bucket assignment logic
        self.buckets = {f"bucket_{i}": i for i in range(self.n_buckets)}
    
    def _assign_buckets(self, data):
        """
        Time-regime aware bucket assignment.
        Buckets are assigned RELATIVE to all plays in the same timestep range.
        """

        n = data.shape[0]
        scores = np.zeros(n)

        # --------------------------------------------------
        # 1. Determine time regime
        # --------------------------------------------------
        duration = self.end - self.start

        # Assume normalized game clock in [0, 1]
        if self.end <= 0.25:
            phase = "early"
        elif self.end <= 0.55:
            phase = "mid"
        elif self.end <= 0.85:
            phase = "late"
        else:
            phase = "critical"

        # --------------------------------------------------
        # 2. Phase-specific scoring
        # --------------------------------------------------
        for i, row in enumerate(data):
            score_diff = self._get_feature(row, "score_difference")
            rel_strength = self._get_feature(row, "relative_strength")
            yards_to_endzone = self._get_feature(row, "end.yardsToEndzone")
            home_possession = self._get_feature(row, "home_has_possession")

            # Base pressure (home perspective)
            pressure = -score_diff

            # Field position danger
            if home_possession:
                field = max(0, (40 - yards_to_endzone) / 40)
            else:
                field = max(0, (60 - yards_to_endzone) / 30)

            # ---------------- Phase modulation ----------------
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

            else:  # critical
                pressure *= 1.6
                pressure += 1.5 * field
                pressure += 1.2 * (not home_possession)

            scores[i] = pressure

        # --------------------------------------------------
        # 3. Distribution-aware bucketing (per timestep)
        # --------------------------------------------------
        qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])

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


    
    def _get_feature(self, row, name):
        """Helper to safely extract a feature value."""
        if name in self.feature_map:
            return row[self.feature_map[name]]
        return None
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return a score matrix where each column is a bucket.
        Score is 1.0 for the assigned bucket, 0.0 for others.
        """
        if len(self.buckets) == 0:
            return np.zeros((X.shape[0], 0))
        
        scores = np.zeros((X.shape[0], self.n_buckets))
        assignments = self._assign_buckets(X)
        
        for i, bucket_id in enumerate(assignments):
            scores[i, bucket_id] = 1.0
        
        return scores