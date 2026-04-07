import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def run_inference(df: pd.DataFrame, model, features: list, column_name: str) -> pd.DataFrame:
    """
    Runs inference on a DataFrame using only the specified feature columns,
    and returns the DataFrame with predictions appended as a new column.

    Parameters
    ----------
    df : pd.DataFrame
    model : fitted sklearn model with a predict() method
    features : list of str
        Column names to extract as input features, in the expected order.
    column_name : str
        Name of the new column to add with the predictions.

    Returns
    -------
    pd.DataFrame with an added column `column_name` containing predictions
    """
    X = df[features].to_numpy()
    df = df.copy()
    df[column_name] = model.predict(X)
    return df


def load_dataset(directory: str, features: list) -> dict:
    """
    Loads all CSVs from every year subfolder in `directory`, keeps only the
    given feature columns, and returns a dict mapping each year to a numpy array.

    Parameters
    ----------
    directory : str
        Root dataset directory (e.g. '.../dataset_interpolated_fixed').
    features : list of str
        Column names to keep, in the desired output order.

    Returns
    -------
    dict[int, np.ndarray]
        Keys are years (int), values are arrays of shape (n_rows, len(features)).
    """
    year_dirs = sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.isdigit()
    )
    if not year_dirs:
        raise FileNotFoundError(f"No year subdirectories found in {directory}")

    result = {}
    for year_str in year_dirs:
        year_dir = os.path.join(directory, year_str)
        dfs = []
        for fname in sorted(os.listdir(year_dir)):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(year_dir, fname))
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise KeyError(f"{fname} (year {year_str}) is missing columns: {missing}")
            dfs.append(df[features])

        if dfs:
            separator = np.zeros((1, len(features)))
            separated = [arr for df in dfs for arr in (df[features].to_numpy(), separator)]
            result[int(year_str)] = np.concatenate(separated[:-1], axis=0)  # drop trailing separator

    return result


def annotate_possessions(data: dict, possession_index: int, score_difference_index: int) -> dict:
    """
    Adds a column to each year's array indicating how many points the team
    currently in possession scores during that drive.

    A drive is a contiguous sequence of rows sharing the same possession value
    (1 = home, 0 = away). The points scored for the drive is:
        delta = score_difference[last_row] - score_difference[first_row]
    Every row in the drive gets this same delta value.
    Separator rows (all zeros) get 0.

    Parameters
    ----------
    data : dict[int, np.ndarray]
        Output of load_dataset — keys are years, values are 2D arrays.
    possession_index : int
        Column index for the possession indicator (1 = home, 0 = away).
    score_difference_index : int
        Column index for score difference (home_score - away_score).

    Returns
    -------
    dict[int, np.ndarray]
        Same structure with one extra column appended.
    """
    result = {}
    for year, arr in data.items():
        n_rows = arr.shape[0]
        points_scored = np.zeros(n_rows)

        i = 0
        while i < n_rows:
            # Separator row — leave as 0 and skip
            if np.all(arr[i] == 0):
                i += 1
                continue

            possession = arr[i, possession_index]
            drive_start = i

            # Advance until possession changes, a separator is hit, or end of array
            while i < n_rows and not np.all(arr[i] == 0) and arr[i, possession_index] == possession:
                i += 1

            drive_end = i - 1
            delta = arr[drive_end, score_difference_index] - arr[drive_start, score_difference_index]
            if drive_start < drive_end:
                points_scored[drive_start:drive_end] = delta

        result[year] = np.concatenate([arr, points_scored.reshape(-1, 1)], axis=1)

    return result


def train_possession_model(annotated_data: dict, train_years: list):
    """
    Trains a linear regression model on the given years, using all columns
    except the last as features and the last column as the target label.

    Parameters
    ----------
    annotated_data : dict[int, np.ndarray]
        Output of annotate_possessions.
    train_years : list of int
        Years to include in training.

    Returns
    -------
    model : fitted LinearRegression
    metrics : dict with 'train_r2' and 'test_r2'
    """
    arrays = [annotated_data[y] for y in train_years if y in annotated_data]
    if not arrays:
        raise ValueError(f"None of the requested train_years found in data: {train_years}")

    combined = np.concatenate(arrays, axis=0)
    X = combined[:, :-1]
    y = combined[:, -1]
    return run_linear_regression(X, y)


def run_logistic_regression(X: np.ndarray, y: np.ndarray, test_size: float = 0.05, random_state: int = 42):
    """
    Fits a logistic regression model on X and y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) — binary or multiclass labels
    test_size : fraction of data held out for evaluation
    random_state : seed for reproducibility

    Returns
    -------
    model : fitted LogisticRegression
    metrics : dict with 'train_accuracy' and 'test_accuracy'
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
    }
    return model, metrics


def run_linear_regression(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Fits a linear regression model on X and y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) — continuous target values
    test_size : fraction of data held out for evaluation
    random_state : seed for reproducibility

    Returns
    -------
    model : fitted LinearRegression
    metrics : dict with 'train_r2' and 'test_r2'
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
    }
    return model, metrics


def run_pipeline(
    directory: str,
    train_years: list,
    test_years: list,
    features: list,
    possession_index: int,
    score_difference_index: int,
    prediction_column: str = "predicted_drive_points",
):
    """
    Full pipeline:

    Train years (leave-one-out):
      For each year Y in train_years, train a model on all other train years,
      then run inference on every CSV in Y and save it in-place with a new column.

    Test years:
      Train a model on all train_years combined, then run inference on every
      CSV in each test year and save it in-place with a new column.

    Parameters
    ----------
    directory : str
        Root dataset directory containing year subfolders.
    train_years : list of int
    test_years : list of int
    features : list of str
        Feature columns used for both training and inference.
    possession_index : int
        Index into `features` for the possession column.
    score_difference_index : int
        Index into `features` for the score difference column.
    prediction_column : str
        Name of the new column written to each CSV.
    """
    # --- Load and annotate all train years ---
    all_train_data = load_dataset(directory, features)
    all_train_data = {y: all_train_data[y] for y in train_years if y in all_train_data}
    annotated = annotate_possessions(all_train_data, possession_index, score_difference_index)

    # --- Leave-one-out inference on train years ---
    for year in train_years:
        if year not in annotated:
            print(f"Skipping train year {year}: not found in dataset")
            continue

        other_train_years = [y for y in train_years if y != year and y in annotated]
        if not other_train_years:
            print(f"Skipping train year {year}: no other years to train on")
            continue

        model, metrics = train_possession_model(annotated, other_train_years)
        print(f"Train year {year} | model trained on {other_train_years} | {metrics}")

        year_dir = os.path.join(directory, str(year))
        for fname in sorted(os.listdir(year_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(year_dir, fname)
            df = pd.read_csv(fpath)
            df = run_inference(df, model, features, prediction_column)
            df.to_csv(fpath, index=False)

    # --- Train on all train years, run inference on test years ---
    model, metrics = train_possession_model(annotated, list(annotated.keys()))
    print(f"Final model trained on {list(annotated.keys())} | {metrics}")

    for year in test_years:
        year_dir = os.path.join(directory, str(year))
        if not os.path.isdir(year_dir):
            print(f"Skipping test year {year}: directory not found")
            continue

        for fname in sorted(os.listdir(year_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(year_dir, fname)
            df = pd.read_csv(fpath)
            df = run_inference(df, model, features, prediction_column)
            df.to_csv(fpath, index=False)


if __name__ == "__main__":
    DIRECTORY = "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/ML/dataset_interpolated_fixed"
    TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    TEST_YEARS = [2023, 2024]
    FEATURES = ["game_completed", "relative_strength", "score_difference", "home_has_possession", "end.down", "end.distance", "end.yardsToEndzone",  "home_timeouts_left", "away_timeouts_left"]
    POSSESSION_INDEX = FEATURES.index("home_has_possession")
    SCORE_DIFFERENCE_INDEX = FEATURES.index("score_difference")

    run_pipeline(
        directory=DIRECTORY,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        features=FEATURES,
        possession_index=POSSESSION_INDEX,
        score_difference_index=SCORE_DIFFERENCE_INDEX,
        prediction_column="predicted_drive_points",
    )
