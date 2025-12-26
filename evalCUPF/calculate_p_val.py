from evalCUPF.entries import Entries
from evalCUPF.C_estimator import estimate_C
import numpy as np
import pandas as pd

B = 10000

def generate_GP(mean_t, cov_matrix: np.ndarray, time_grid: np.ndarray):
    """Generates Gaussian process with given mean and covariance
    Args:
    mean_t: function of t
    cov_matrix: square matrix

    """
    mean_vec = np.array([mean_t(t) for t in time_grid])
    rng = np.random.default_rng()
    try:
        # try Cholesky first (fast)
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # fallback: eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        print(eigvals)
        eigvals[eigvals < 0] = 0.0    # clip small negatives
        L = eigvecs * np.sqrt(eigvals)[None, :]
    z = rng.normal(size=(cov_matrix.shape[0],))
    return L @ z + mean_vec

def brier_loss(y_true: np.ndarray, y_pred: np.ndarray): 
        """Calculates Brier loss for matrices (mean squared error between predictions and binary outcomes)
        Args:
            y_true: binary outcomes (0 or 1)
            y_pred: predicted probabilities (between 0 and 1)
        
        Returns:
            float: mean squared error
        """
        return (y_pred - y_true) ** 2

def calculate_delta(entries: Entries, loss_fn = brier_loss) -> np.ndarray:
      """
      Calculate the loss average between forecasts A and B.
      Returns a NumPy array which corresponds the delta at each timestep.
      """
      pA = entries.p_A
      pB = entries.p_B
      Y = entries.Y
      print(pA.shape, pB.shape, Y.shape)
      loss_diff = loss_fn(Y, pA) - loss_fn(Y, pB)
      return loss_diff.mean(axis=0)

def calculate_p_val(entries: Entries, B = B):
    # Generate B GPs
    covariance_matrix = estimate_C(entries, None) # NEED TO CHECK IF POSITIVE SEMIDEFINITE
    n = entries.n
    # Compute T_n
    delta_n = calculate_delta(entries, brier_loss)
    sqrt_n_delta = np.sqrt(n) * delta_n
    T_n = np.max(np.abs(sqrt_n_delta))
    p = 0
    for i in range(B):
        if i % 1000 == 0:
            print(i)
        gp_sample = generate_GP(lambda t: 0, covariance_matrix, entries.timesteps)
        sup_gp = np.max(np.abs(gp_sample))
        if T_n >= sup_gp:
             p += 1
    p_val = 1 - p / B
    return p_val

if __name__ == "__main__":
    entries = Entries(0.005)
    file_path = "NFL/test_7/ensemble_model_testing_2_combined_data.csv"
    df = pd.read_csv(file_path)
    entries.load_entries(df, "game_completed", "phat_A", "phat_B", "Y", "game_id")
    p_val = calculate_p_val(entries)
    print(p_val)