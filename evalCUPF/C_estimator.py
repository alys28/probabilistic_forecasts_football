from evalCUPF.entries import Entries
from typing import List
import numpy as np

def C_cons(pA, pB):
    X = pA-pB # (n_games, n_timesteps)
    delta_bar = np.mean(X, axis = 0) # (n_timesteps,)
    X_centered = X - delta_bar[None, :] # Broadcast to (1, n_timesteps) -> (n_games, n_timesteps)
    # Compute kernel for each entry (i, j) = C(i, j)
    n_games = pA.shape[0]
    C_hat = (X_centered.T @ X_centered) / n_games    # shape (n_timesteps, n_timesteps)
    return C_hat

def estimate_C(entries: Entries, p_est: List = None):
    """
    Outputs the covariance matrix, built from the entries of 'entries', using the given method to estimate ground truth probabilities. 
    Args:
    - p_est: ground truth probabilities if we use a different estimate from 1/4
    Output:
    Covariance matrix of shape (n_timesteps, n_timesteps), where n_timesteps is the 2nd axis's dimension in entries.
    """
    pA = entries.p_A
    pB = entries.p_B
    if p_est is None:
        return C_cons(pA, pB)
    raise NotImplementedError("Currently only supporting a conservative estimate of C.")


def estimate_buckets():
    pass
    
