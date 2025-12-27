from evalCUPF.risk_buckets import Bucketer, bucket_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NFLBucketer(Bucketer):
    def __init__(self, features, data, start, end, n_buckets=3, random_state=42):
        """
        n_buckets: number of buckets
        """
        self.n_buckets = n_buckets
        self.random_state = random_state
        super().__init__(features, data, start, end)
    
    def _preprocess_strategy(self, data):
        """
        Use K-means to define buckets (clusters) in the feature space.
        Features are scaled and cosine similarity will be used for scoring.
        """
        if data.size == 0:
            self.buckets = {}
            self.v = {}
            return

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data)

        # K-means clustering
        self.kmeans = KMeans(n_buckets=self.n_buckets, random_state=self.random_state)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Store buckets: each bucket is represented by its cluster centroid
        self.buckets = {f"bucket_{i}": self.kmeans.cluster_centers_[i] for i in range(self.n_buckets)}
        # Get the unbiased estimate of p(1-p), as described in https://arxiv.org/pdf/1202.5140
        for j in range(self.n_buckets):
            n_j_t = np.sum(cluster_labels == j)
            self.v = {f"bucket_{i}": np.sum(cluster_labels == i) for i in range(self.n_buckets)}

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return a score for each row for each bucket using cosine similarity
        """
        if len(self.buckets) == 0:
            return np.zeros((X.shape[0], 0))

        X_scaled = self.scaler.transform(X)
        bucket_names = list(self.buckets.keys())
        centroids = np.array([self.buckets[b] for b in bucket_names])

        # Cosine similarity between each row and each centroid
        scores = cosine_similarity(X_scaled, centroids)
        return scores
