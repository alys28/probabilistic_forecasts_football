import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_methods.siamese_network import SiameseClassifier, SiameseNetwork, ContrastiveLoss
from notebooks import process_data
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def setup_models(features_data, features, num_models = 10, epochs = 100, lr = 0.0001, batch_size = 128, hidden_dim = 128):
    """
    Setup models for each timestep range.
    """
    # Setup models for each timestep range
    models = []
    for i in range(num_models):
        timesteps_range = [0.97, 1]
        X = []
        y = []
        for timestep in features_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                X.extend(np.array(features_data[timestep])[:,2:])
                y.extend(np.array(features_data[timestep])[:,1])
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        if len(X) == 0 or len(y) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
        
        # Split data: 80% training, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify by y to ensure equal distribution of classes in both sets
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
            
        # Concatenate all data points from different timesteps in this range
        print("Training for timestep", timesteps_range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug: Print the actual input dimension being used
        actual_input_dim = X_train.shape[1]
        expected_input_dim = len(features) - 2
        print(f"Actual input dimension: {actual_input_dim}")
        print(f"Expected input dimension: {expected_input_dim}")
        
        # Use the actual input dimension instead of len(features) - 1
        siamese_network = SiameseNetwork(actual_input_dim, hidden_dim)
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = torch.optim.Adam(siamese_network.parameters(), lr=lr)
        siamese_classifier = SiameseClassifier(siamese_network, epochs, optimizer, criterion, device)
        
        # Since we remove the first column ([:,1:]), the score difference index is reduced by 1
        # If score difference was originally at index 1, it's now at index 0
        score_diff_index = 0  # Adjust this if your score difference is at a different position
        
        siamese_classifier.fit(X_train, y_train, val_X = X_val, val_y = y_val, batch_size = batch_size, score_difference_index = score_diff_index)
        models.append(siamese_classifier)
        break
    return models