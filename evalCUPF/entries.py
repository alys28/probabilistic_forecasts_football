from dataclasses import dataclass
from typing import List, Any, Union
import pandas as pd
import numpy as np

class Entries:
    """
    Class to hold arrays of probabilities for each timestep in numpy arrays to optimize calculations
    """
    def __init__(self, timestep_size = 0.005):
        self.timesteps = list(range(0, 1, timestep_size))
        self.timestep_size = timestep_size
        self.p_B = None
        self.p_A = None
        self.Y = None
    
    def load_entries(self, data: pd.DataFrame, timestep: str, p_A: str, p_B: str, y: str = "Y", id_field: str = "id"):
        """Load entries from a DataFrame grouped by ID field.
        Assumes the input DataFrame follows the format of the provided sample,
        with each set of forecasts identified by a unique ID. Groups the data
        by ID and initializes probability arrays for teams A and B across timesteps.
        Args:
            data (pd.DataFrame): Input DataFrame containing forecast data across all sequences.
            timestep (str): Column name for timestep values.
            p_A (str): Column name for team A probability values.
            p_B (str): Column name for team B probability values.
            id_field (str, optional): Column name for the ID field. Defaults to "id".
        """
        # Separate the dataframe into smaller frames, based on when the IDs change
        games = [group.reset_index(drop=True) for _, group in data.groupby(id_field)]
        self.p_B = np.zeros((len(games), len(self.timesteps)))
        self.p_A = np.zeros((len(games), len(self.timesteps)))
        self.Y = np.zeros((len(games), len(self.timesteps)))

        for i in range(len(games)):
            df = games[i]
            timestep_col = df[p_A].values
            p_A_col = df[p_A].values
            p_B_col = df[p_B].values
            Y_col = df[y].values
            assert len(len(timestep_col) == len(self.timesteps)), f"Make sure that all your entries in your dataframe have the same number of timesteps and match with the expected number of timesteps, based on the given timestep size: {self.timestep_size}."
            self.p_A[i, :] = p_A_col
            self.p_B[i, :] = p_B_col
            self.Y[i, :] = Y_col

    
    def __getitem__(self, key: Union[tuple[int, int], int]) -> Union[tuple[np.ndarray, np.ndarray], tuple[float, float]]:
        """
        Single number as key[i]: get the forecasts of game with idx i
        Tuple[i, j]: get the forecasts of game i and timestep (j * timestep_size)
        Note: Both indexing methods will return forecasts of both A and B
        """
        assert self.A is not None, "Load entries before indexing."
        if isinstance(key, tuple):
            # Two-index tuple like (i, j) -> both teams scalars
            i, j = key
            a_val = self.p_A[i, j]
            b_val = self.p_B[i, j]
            y_val = self.Y[i, j]
            return (a_val, b_val, y_val)
        else:
            # Single index [i] -> return entire forecast arrays for game i
            a_vals = self.p_A[key, :]
            b_vals = self.p_B[key, :]
            y_vals = self.Y[key, :]
            return (a_vals, b_vals, y_vals)