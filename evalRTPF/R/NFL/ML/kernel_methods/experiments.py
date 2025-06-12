from matplotlib import pyplot as plt
# Import parent directory dynamically
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "notebooks"))
import numpy as np
import pandas as pd
import process_data



# import parent directory
interpolated_dir = os.path.join(parent_dir, "dataset_interpolated_with_overtime")


data = process_data.load_training_data(interpolated_dir, test=[2023, 2024])

other_features = [ "homeScore", "awayScore", "start.down", "start.distance", "start.yardLine",
            "end.down", "end.distance", "end.yardLine",
            "relative_strength", "scoringPlay", "home_has_possession", "home_timeouts_left", "away_timeouts_left"]
label_feature = "home_win"

# Create plots for each timestep
for timestep in data.keys():
    df_list = data[timestep]
    df = pd.DataFrame(df_list)
    print("DataFrame columns:", df.columns)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['homeScore'] - df['awayScore'], df[label_feature], alpha=0.5)
    plt.xlabel('Score Difference (Home - Away)')
    plt.ylabel('Home Win Probability')
    plt.title(f'Score Difference vs Home Win Probability at {timestep:.2%} of Game')
    plt.grid(True)
    plt.show()

models = {}
def setup_models(features_data):
    for timestep in features_data:
        data = np.array(features_data[timestep])
        X_df = pd.DataFrame(data[:, 1:], columns=features[1:]) # Not including home_win (label)
        y = data[:, 0]
        model = create_model()
        print("Training for timestep", timestep)
        model.fit(X_df, y)
        
        # Calculate training loss
        y_pred = model.predict_proba(X_df)[:, 1]  # Get probability predictions
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15))  # Binary cross entropy
        accuracy = model.score(X_df, y)
        print(f"Timestep {timestep:.2%}: Training Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        models[timestep] = model
