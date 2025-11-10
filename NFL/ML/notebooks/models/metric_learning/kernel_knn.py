import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric_learning.siamese_network import SiameseClassifier, SiameseNetwork
from metric_learning.siamese_transformer import SiameseTransformer, SiameseTransformerClassifier
from metric_learning.siamese_lstm import SiameseLSTM, SiameseLSTMClassifier
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from metric_learning.utils import ContrastiveLoss

def setup_models(training_data, test_data, num_models = 20, epochs = 100, lr = 0.001, batch_size = 128, hidden_dim = 128, head_output_dim = 32):
    """
    Setup models for each timestep range with normalization pipeline.
    """
    # Setup models for each timestep range
    models = []
    for i in range(num_models):  # Quick test on one timestep first
        # Divide [0, 1] into num_models equal ranges
        range_size = 1.0 / (num_models - 1)
        start_time = round(i * range_size, 3)
        end_time = round((i + 1) * range_size, 3)
        timesteps_range = [start_time, end_time]
        X = []
        y = []
        X_test = []
        y_test = []
        for timestep in training_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in training_data[timestep]:
                    X.append(np.array(row['rows'], dtype=np.float32))
                    y.append(np.array(row['label'], dtype=np.float32))
        for timestep in test_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in test_data[timestep]:
                    X_test.append(np.array(row['rows'], dtype=np.float32))
                    y_test.append(np.array(row['label'], dtype=np.float32))
        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.float32).reshape(-1, 1)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)
        if len(X) == 0 or len(y) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
        
        # Normalize 3D data (batch_size, seq_len, input_dim) if needed
        # Reshape to 2D for normalization: (batch_size * seq_len, input_dim)
        original_train_shape = X_train.shape
        original_test_shape = X_test.shape
        
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data and transform both train and test
        scaler = StandardScaler()
        X_train_normalized_2d = scaler.fit_transform(X_train_2d)
        X_test_normalized_2d = scaler.transform(X_test_2d)
        
        # Reshape back to original 3D structure
        X_train = X_train_normalized_2d.reshape(original_train_shape)
        X_test = X_test_normalized_2d.reshape(original_test_shape)
        
        # Split data: 80% training, 20% validation
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_test.shape}")
            
        # Concatenate all data points from different timesteps in this range
        print("Training for timestep", timesteps_range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug: Print the actual input dimension being used
        actual_input_dim = X_train.shape[-1]
        print(f"Actual input dimension: {actual_input_dim}")
        print(f"Using device: {device}")
        
        # Use the actual input dimension instead of len(features) - 1
        siamese_network = SiameseNetwork(actual_input_dim, hidden_dim, head_output_dim=head_output_dim)
        criterion = nn.BCELoss()
        siamese_network = siamese_network.to(device)
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(siamese_network.parameters(), lr=lr, weight_decay=0.01)  # Lighter regularization for simpler network
        
        # Add learning rate scheduler - more patience for simpler network
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)
        
        siamese_classifier = SiameseClassifier(siamese_network, epochs, optimizer, criterion, device, scheduler)
        
        # Reduce pairs per sample to prevent overfitting
        siamese_classifier.fit(X_train, y_train, val_X = X_test, val_y = y_test, batch_size = batch_size)
        
        # Save the trained model with informative filename
        model_save_dir = "saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_prefix = os.path.join(model_save_dir, "siamese_model_absolute_diff")
        saved_filepath = siamese_classifier.save_model(model_prefix, timesteps_range)
        print(f"Model {i+1}/{num_models} saved to: {saved_filepath}")
        
        models.append(siamese_classifier)
        
    return models

def setup_transformer_models(training_data, test_data, num_models = 20, epochs = 100, lr = 0.001, batch_size = 128, hidden_dim = 32, head_output_dim = 16):
    """
    Setup models for each timestep range with normalization pipeline.
    """
    # Setup models for each timestep range
    models = []
    for i in range(num_models):  # Quick test on one timestep first
        # Divide [0, 1] into num_models equal ranges
        range_size = 1.0 / (num_models - 1)
        start_time = round(i * range_size, 3)
        end_time = round((i + 1) * range_size, 3)
        timesteps_range = [start_time, end_time]
        X = []
        y = []
        X_test = []
        y_test = []
        for timestep in training_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in training_data[timestep]:
                    X.append(np.array(row['rows'], dtype=np.float32))
                    y.append(np.array(row['label'], dtype=np.float32))
        for timestep in test_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in test_data[timestep]:
                    X_test.append(np.array(row['rows'], dtype=np.float32))
                    y_test.append(np.array(row['label'], dtype=np.float32))
        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.float32).reshape(-1, 1)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)
        if len(X) == 0 or len(y) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
        
        # Normalize 3D data (batch_size, seq_len, input_dim) if needed
        # Reshape to 2D for normalization: (batch_size * seq_len, input_dim)
        original_train_shape = X_train.shape
        original_test_shape = X_test.shape
        
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data and transform both train and test
        scaler = StandardScaler()
        X_train_normalized_2d = scaler.fit_transform(X_train_2d)
        X_test_normalized_2d = scaler.transform(X_test_2d)
        
        # Reshape back to original 3D structure
        X_train = X_train_normalized_2d.reshape(original_train_shape)
        X_test = X_test_normalized_2d.reshape(original_test_shape)
        
        # Split data: 80% training, 20% validation
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_test.shape}")
            
        # Concatenate all data points from different timesteps in this range
        print("Training for timestep", timesteps_range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug: Print the actual input dimension being used
        actual_input_dim = X_train.shape[-1]
        print(f"Actual input dimension: {actual_input_dim}")
        print(f"Using device: {device}")
        
        # Use the actual input dimension instead of len(features) - 1
        siamese_network = SiameseTransformer(actual_input_dim, hidden_dim, head_output_dim=head_output_dim)
        criterion = nn.BCELoss()
        siamese_network = siamese_network.to(device)
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(siamese_network.parameters(), lr=lr, weight_decay=0.01)  # Lighter regularization for simpler network
        
        # Add learning rate scheduler - more patience for simpler network
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)
        
        siamese_classifier = SiameseTransformerClassifier(siamese_network, epochs, optimizer, criterion, device, scheduler)
        
        # Reduce pairs per sample to prevent overfitting
        siamese_classifier.fit(X_train, y_train, val_X = X_test, val_y = y_test, batch_size = batch_size)
        
        # Save the trained model with informative filename
        model_save_dir = "saved_models_transformer"
        os.makedirs(model_save_dir, exist_ok=True)
        model_prefix = os.path.join(model_save_dir, "siamese_model")
        saved_filepath = siamese_classifier.save_model(model_prefix, timesteps_range)
        print(f"Model {i+1}/{num_models} saved to: {saved_filepath}")
        
        models.append(siamese_classifier)
        
    return models

def setup_lstm_models(training_data, test_data, num_models = 20, epochs = 100, lr = 0.0001, batch_size = 128, hidden_dim = 32, head_output_dim = 16, lstm_layers = 1, sequence_length = 1):
    """
    Setup LSTM models for each timestep range with normalization pipeline.
    
    Args:
        sequence_length: Length of sequence. If data is 2D (n_samples, n_features), 
                        it will be reshaped to (n_samples, sequence_length, n_features/sequence_length).
                        Set to 1 to treat each row as a single timestep.
    """
    # Setup models for each timestep range
    models = []
    for i in range(67, num_models):
        # Divide [0, 1] into num_models equal ranges
        range_size = 1.0 / (num_models - 1)
        start_time = round(i * range_size, 3)
        end_time = round((i + 1) * range_size, 3)
        timesteps_range = [start_time, end_time]
        X = []
        y = []
        X_test = []
        y_test = []
        for timestep in training_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in training_data[timestep]:
                    X.append(np.array(row['rows'], dtype=np.float32))
                    y.append(np.array(row['label'], dtype=np.float32))
        for timestep in test_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in test_data[timestep]:
                    X_test.append(np.array(row['rows'], dtype=np.float32))
                    y_test.append(np.array(row['label'], dtype=np.float32))
        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.float32).reshape(-1, 1)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)
        if len(X) == 0 or len(y) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
        
        # Normalize 2D data
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data and transform both train and test
        scaler = StandardScaler()
        X_train_normalized_2d = scaler.fit_transform(X_train_2d)
        X_test_normalized_2d = scaler.transform(X_test_2d)
        
        # Reshape back to original structure
        X_train = X_train_normalized_2d.reshape(X_train.shape)
        X_test = X_test_normalized_2d.reshape(X_test.shape)
        
        # Reshape for LSTM: (n_samples, n_features) -> (n_samples, sequence_length, n_features/sequence_length)
        # If sequence_length=1, treat each feature vector as a single timestep
        if sequence_length == 1:
            # Reshape to (n_samples, 1, n_features)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_test.shape}")
        # Concatenate all data points from different timesteps in this range
        print("Training for timestep", timesteps_range)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug: Print the actual input dimension being used
        actual_input_dim = X_train.shape[-1]
        print(f"Actual input dimension: {actual_input_dim}")
        print(f"Using device: {device}")
        
        # Create LSTM-based Siamese network
        siamese_lstm = SiameseLSTM(
            input_dim=actual_input_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            head_output_dim=head_output_dim,
            dropout_rate=0.3,
            bidirectional=False
        )
        criterion = ContrastiveLoss(margin = 0.5)
        siamese_lstm = siamese_lstm.to(device)
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(siamese_lstm.parameters(), lr=lr, weight_decay=1e-4)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)
        
        siamese_classifier = SiameseLSTMClassifier(
            siamese_lstm, epochs, optimizer, criterion, device, scheduler, 
            sequence_length=sequence_length
        )
        
        # Reduce pairs per sample to prevent overfitting
        siamese_classifier.fit(X_train, y_train, val_X = X_test, val_y = y_test, batch_size = batch_size)
        
        # Save the trained model with informative filename
        model_save_dir = "saved_models_lstm"
        os.makedirs(model_save_dir, exist_ok=True)
        model_prefix = os.path.join(model_save_dir, "siamese_model_LSTM")
        saved_filepath = siamese_classifier.save_model(model_prefix, timesteps_range)
        print(f"Model {i+1}/{num_models} saved to: {saved_filepath}")
        
        models.append(siamese_classifier)
        
    return models