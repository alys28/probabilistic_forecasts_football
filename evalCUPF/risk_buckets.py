from abc import ABC, abstractmethod
import numpy as np
class Bucketer(ABC):
    """
    Abstract class for a risk bucket strategy. One Bucketer object must be created per timestep.
    """
    def __init__(self):
        self.buckets = {}
        self.v = {} # estimator for each bucket
        self._preprocess_strategy()

    @abstractmethod
    def _preprocess_strategy(self):
        """
        Strategy to organize given data into bucket groups.
        """
    
    def assign_bucket(self, X: np.ndarray, return_v: bool = True) -> np.ndarray:
        """
        Call the score method, then keep the bucket with max score
        Args:
        X: 2D input array
        return_v: should return estimator, otherwise return bucket name
        Return: 1D output array, where each entry at idx i is the best fit bucket for the input at idx i.
        """
        scores = self.score(X)
        best_bucket_indices = np.argmax(scores, axis=1)
        bucket_names = list(self.buckets.keys())
        best_buckets = np.array([bucket_names[i] for i in best_bucket_indices])
        if return_v:
            return np.array([self.v[bucket] for bucket in best_buckets])
        return best_buckets
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get comparison score for each bucket
        Args:
        X: 2D input array
        Return: 2D output array, where each entry at idx i is an array of scores of each bucket for the input at idx i.
        """
    
class BucketContainer:
    def __init__(self):
        self._bucketers = {}

    def __getitem__(self, key):
        return self._bucketers[key]

    def __setitem__(self, key, value: Bucketer):
        self._bucketers[key] = value
    
    def assign_bucket(self, X: np.ndarray, t, return_v: bool = True) -> np.ndarray:
        assert t in self._bucketers.keys(), "No buckets available for given timestep. Initialize the bucket first."
        return self._bucketers[t].assign_bucket(X, return_v)

def process_data(dir, features) -> BucketContainer:
    """

    
    """
    # Takes a directory as a parameter, loads CSVs and creates np array for each timestep. Then for each array, create a Bucketer object. Then we are done for this. We just have to add an estimator function in C_estimator.py and do matmul.
