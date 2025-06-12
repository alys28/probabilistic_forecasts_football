import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_training_data(interpolated_dir, test = [2023, 2024]):
    training_data = {}
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if not(int(folder) in test):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        for _, row in df.iloc[1:].iterrows():
                            row["relative_strength"] = df.iloc[0]["homeWinProbability"]
                            row["away_team_id"] = df.iloc[0]["away_team_id"]
                            row["home_team_id"] = df.iloc[0]["home_team_id"]
                            row["home_win"] = df.iloc[0]["home_win"]
                            if row["timestep"] not in training_data:
                                training_data[row["timestep"]] = [row] 
                            else:
                                training_data[row["timestep"]] += [row]
                        
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

def plot_accuracy(models, tests, title=""):
    # Test accuracy of model for each timestep on test data and plot
    accuracies = []
    timesteps = []

    for timestep, i in zip(models, tests.keys()):
        model = models[timestep]
        test = np.array(tests[i])
        y_test = test[:,0]
        X_test = test[:,1:]
        accuracy = model.score(X_test, y_test)
        
        accuracies.append(accuracy)
        timesteps.append(timestep)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, accuracies, color='tab:blue', label='Accuracy')
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(models, tests, title=""):
    # Test BCE loss of model for each timestep on test data and plot
    losses = []
    timesteps = []

    for timestep, i in zip(models, tests.keys()):
        model = models[timestep]
        test = np.array(tests[i])
        y_test = test[:,0]
        X_test = test[:,1:]
        
        # Calculate BCE loss
        y_pred = model.predict_proba(X_test)[:, 1]
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

def write_predictions(models, features_test_data, interpolated_dir, phat_b = "phat_b"):
# Write the predictions to csv file
    for folder in features_test_data:
        print(folder)
        test_data = features_test_data[folder]
        
        for file in test_data:
            df = pd.read_csv(os.path.join(interpolated_dir, folder, file))

            # Precompute rounded timesteps for faster lookup
            df["rounded_timestep"] = df["timestep"].round(3)
            rows = df.iloc[1:].iterrows()
            index, row = next(rows)
            for i, timestep in enumerate(models):
                model = models[timestep]
                X_test = np.array(test_data[file][i])[1:].reshape(1, -1)
                pred = model.predict_proba(X_test)[0][1]
                try:
                    while round(row["timestep"], 3) == round(timestep, 3):
                        df.at[index, phat_b] = pred
                        index, row = next(rows)
                except StopIteration:
                    pass
            # Save the file once after all updates
            df.drop(columns=["rounded_timestep"], inplace=True)
            df.to_csv(os.path.join(interpolated_dir, folder, file), index=False)
            print(f"Finished writing to {file}")
