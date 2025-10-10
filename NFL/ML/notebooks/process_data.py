import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

def load_data(interpolated_dir, years, history_length, features, label_feature, replace_nan_val = 0, train = True):
    training_data = defaultdict(list)
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if (int(folder) in years):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        df.loc[1:, "relative_strength"] = df.iloc[0]["homeWinProbability"]
                        df.loc[1:, "away_team_id"] = df.iloc[0]["away_team_id"]
                        df.loc[1:, "home_team_id"] = df.iloc[0]["home_team_id"]
                        df.loc[1:, "home_win"] = df.iloc[0]["home_win"]
                        for idx in range(1, len(df)):
                            current_row = df.iloc[idx]
                            label = current_row[label_feature]
                            try:
                                label_float = float(label)
                                if np.isnan(label_float):
                                    print(f"  NaN Label found in file: {file_path}")
                                    break
                            except (ValueError, TypeError):
                                print(f"  Invalid label in file: {file_path}")
                            
                            # Check for NaN in current row features (safer method)
                            try:
                                current_row_features = current_row[features].to_numpy(dtype=np.float32)
                                if np.isnan(current_row_features).any():
                                    print(f"  NaN found in file: {file_path}")
                                    current_row_features = np.nan_to_num(current_row_features, nan=0.0)
                            except (ValueError, TypeError):
                                print(f"  Invalid features in file: {file_path}")
                            current_row_np = current_row_features.reshape(1, -1)
                            start_idx = max(1, idx - history_length)
                            actual_history_len = idx - start_idx
                            
                            # Check for NaN in history rows (safer method)
                            if history_length > 0:
                                history_rows = df.iloc[start_idx:idx][features].to_numpy(dtype=np.float32)
                                if np.isnan(history_rows).any():
                                    print(f"  NaN found in file: {file_path}")
                                    history_rows = np.nan_to_num(history_rows, nan=0.0)
                                
                                if actual_history_len < history_length:
                                    padding = np.zeros((history_length - actual_history_len, len(features)))
                                    history_rows = np.concatenate([padding, history_rows], axis=0)
                                
                                final_rows_for_timestep = np.concatenate([history_rows, current_row_np], axis=0)
                            else:
                                final_rows_for_timestep = current_row_np.reshape(-1)
                            if train:
                                training_data[round(current_row["model"], 3)].append({"rows": final_rows_for_timestep, "label": label_float})
                            else:
                                training_data[round(current_row["timestep"], 3)].append({"rows": final_rows_for_timestep, "label": label_float, "model": round(current_row["model"], 3)})
                     
            else: 
                print("skipping ", folder)
    return training_data

def feature_selection(data, features, replace_nan_val = 0):
    # Given the features of the data, return data such that each row is an array of the values of the features
    # The data is a dictionary where the key is the timestep and the value is a list of rows
    feature_data = {}
    for timestep in data:
        feature_data[timestep] = []
        for row in data[timestep]:
            new_row = [float(row[feature]) for feature in features]
            # First check if the row has any NaN values
            new_row = [val if not np.isnan(val) else replace_nan_val for val in new_row] 
            feature_data[timestep].append(new_row)
    return feature_data


def setup_models(features_data, MODEL, *args, **kwargs):
    models = {}
    for timestep in features_data:
        X = np.array(features_data[timestep])[:,1:]
        y = np.array(features_data[timestep])[:,0]
        model = MODEL(*args, **kwargs)
        
        print("Training for timestep", timestep)
        model.fit(X, y)

        # Calculate training loss
        y_pred = model.predict_proba(X)[:, 1]  # Get probability predictions
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15))  # Binary cross entropy
        accuracy = model.score(X, y)
        print(f"Timestep {timestep:.2%}: Training Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        models[timestep] = model
    return models

def setup_models_DL(features_data, MODEL, *args, **kwargs):
    '''
    Converts features into torch tensors and trains the model
    '''
    models = {}
    for timestep in features_data:
        X = torch.tensor(np.array(features_data[timestep])[:, 1:], dtype=torch.float32)
        y = torch.tensor(np.array(features_data[timestep])[:, 0])
        model = MODEL(*args, **kwargs)
        print("Training for timestep", timestep)
        model.fit(X, y)
        models[timestep] = model
    return models

# DEPRECATED
def load_test_data(interpolated_dir, test = [2023, 2024]):
    test_folders = {}
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if int(folder) in test:
                test_data = {}
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        seen_timesteps = set()
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        for _, row in df.iloc[1:].iterrows():
                            row["relative_strength"] = df.iloc[0]["homeWinProbability"]
                            row["away_team_id"] = df.iloc[0]["away_team_id"]
                            row["home_team_id"] = df.iloc[0]["home_team_id"]
                            row["home_win"] = df.iloc[0]["home_win"]
                            if file not in test_data and row["timestep"] not in seen_timesteps:
                                seen_timesteps.add(row["timestep"])
                                test_data[file] = [row] 
                            elif row["timestep"] not in seen_timesteps:
                                seen_timesteps.add(row["timestep"]) 
                                test_data[file] += [row]
                test_folders[folder] = test_data
    return test_folders

# DEPRECATED
def test_feature_selection(data, features, replace_nan_val = 0):
    # Given the features of the data, return data such that each row is an array of the values of the features
    # The data is a dictionary where the key is the timestep and the value is a list of rows
    feature_data = {}
    for file in data:
        feature_data[file] = []
        for row in data[file]:
            new_row = [float(row[feature]) for feature in features]
            # First check if the row has any NaN values
            new_row = [val if not np.isnan(val) else replace_nan_val for val in new_row] 
            feature_data[file].append(new_row)
    return feature_data

def plot_accuracy(models, test_data, title=""):
    f"""
    Test accuracy of model for each timestep on test data and plot
    Args:
    - models: Dict[float: MODEL_TYPE] where the keys are timesteps between 0 and 1
    - test_data: Dict[float: Dict[str: np.array, str: float, model: float]], coming from load_data (with train = False)
    - (Optional) title: Plot Title
    """
    accuracies = []
    timesteps = []
    for timestep in test_data.keys():
        correct = 0
        total = 0
        for entry in timestep:
            X_test = entry["rows"]
            y_test = entry["label"]
            model = models[entry["model"]]
            accuracy = model.score(X_test, y_test) # will be either 0 or 1 (since it's only 1 entry tested at a time)
            total += 1
            correct = accuracy
        accuracies.append(correct / total)
        timesteps.append(timestep)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, accuracies, color='tab:blue', label='Accuracy')
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(models, test_data, title=""):
    # Test BCE loss of model for each timestep on test data and plot
    losses = []
    timesteps = []
    for timestep in test_data.keys():
        y_preds = []
        y_tests = []
        for entry in timestep:
            X_test = entry["rows"]
            y_test = entry["label"]
            model = models[entry["model"]]
            # Calculate BCE loss
            y_pred = model.predict_proba(X_test)[:, 1]
            y_preds.append(y_pred)
            y_tests.append(y_test)
        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)
        loss = -np.mean(y_test * np.log(y_pred + 1e-15) + (1-y_test) * np.log(1-y_pred + 1e-15))
        
        losses.append(loss)
        timesteps.append(timestep)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, losses, color='tab:red', label='BCE Loss')
    plt.xlabel('Timestep')
    plt.ylabel('BCE Loss')
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Keep the original plot function for backward compatibility
def plot(models, tests, title=""):
    plot_accuracy(models, tests, title)
    plot_loss(models, tests, title)

def write_predictions(models, interpolated_dir, years, history_length, features, replace_nan_val = 0, phat_b = "phat_b"):
    # Write the predictions to csv file
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        if os.path.isdir(folder_path):
            if (int(folder) in years):
                print(f"Loading data for {folder}")
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        df.loc[1:, "relative_strength"] = df.iloc[0]["homeWinProbability"]
                        df.loc[1:, "away_team_id"] = df.iloc[0]["away_team_id"]
                        df.loc[1:, "home_team_id"] = df.iloc[0]["home_team_id"]
                        df.loc[1:, "home_win"] = df.iloc[0]["home_win"]
                        for idx in range(1, len(df)):
                            current_row = df.iloc[idx]
                            # Get column names
                            current_row_features = current_row[features].to_numpy(dtype=np.float32)
                            current_row_features = np.nan_to_num(current_row_features, nan=replace_nan_val)
                            
                            current_row_np = current_row_features.reshape(1, -1)
                            start_idx = max(1, idx - history_length)
                            actual_history_len = idx - start_idx
                            if history_length > 0:
                                history_rows = df.iloc[start_idx:idx][features].to_numpy(dtype=np.float32)
                                history_rows = np.nan_to_num(history_rows, nan=replace_nan_val)
                                if actual_history_len < history_length:
                                    padding = np.zeros((history_length - actual_history_len, len(features)))
                                    history_rows = np.concatenate([padding, history_rows], axis=0)
                                final_rows_for_timestep = np.concatenate([history_rows, current_row_np], axis=0)
                            else:
                                final_rows_for_timestep = current_row_np.reshape(-1)
                            # Do inference
                            model_assigned = round(current_row["model"], 3)
                            model = models[model_assigned]
                            X_test = np.expand_dims(final_rows_for_timestep, axis=0)
                            pred = model.predict_proba(X_test)[0][1]
                            df.at[idx, phat_b] = pred
                        df.to_csv(file_path, index=False)
                        print("Processed file: ", file)
