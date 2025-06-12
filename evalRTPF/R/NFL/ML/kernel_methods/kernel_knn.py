import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from siamese_network import SiameseClassifier, SiameseNetwork
from notebooks import process_data
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

interpolated_dir = os.path.join(parent_dir, "dataset_interpolated_with_overtime")
training_data = process_data.load_training_data(interpolated_dir)
features = ["home_win", "relative_strength", "homeScore", "awayScore", "scoringPlay", "start.down", "start.distance", "start.yardLine", "end.down", "end.distance", "end.yardLine", "home_timeouts_left", "away_timeouts_left", "home_has_possession"]

count = 0
for timestep in training_data.keys():
    rows = training_data[timestep]
    
    # Replace NaN values with 0
    for row in rows:
        for feature in features:
            if np.isnan(row[feature]):
                count += 1
                row[feature] = 0

print("Number of NaN values:", count)

features_data = process_data.feature_selection(training_data, features)



def setup_models(features_data, num_models = 20, epochs = 100, lr = 0.001, batch_size = 128, hidden_dim = 100):
    """
    Setup models for each timestep range.
    """
    # Setup models for each timestep range
    models = []
    for i in range(num_models):
        timesteps_range = [i/num_models, (i+1)/num_models]
        X = []
        y = []
        for timestep in features_data:
            if timestep[0] >= timesteps_range[0] and timestep[1] < timesteps_range[1]:
                X.append(np.array(features_data[timestep])[:,1:])
                y.append(np.array(features_data[timestep])[:,0])
        X = np.array(X)
        y = np.array(y)
        print("Training for timestep", timesteps_range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        siamese_network = SiameseNetwork(len(features) - 1, hidden_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(siamese_network.parameters(), lr=lr)
        siamese_classifier = SiameseClassifier(siamese_network, epochs, optimizer, criterion, device)
        siamese_classifier.fit(X, y, val_X = None, val_y = None, batch_size = batch_size)
        models.append(siamese_classifier)
    return models
