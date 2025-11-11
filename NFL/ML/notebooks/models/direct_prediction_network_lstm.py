import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
from .DL_Model import BaseDirectClassifier, NFLDirectDataset

# Custom Brier Score Loss Function
class BrierLoss(nn.Module):
    """
    Brier Score Loss for binary classification.
    Lower values indicate better probabilistic predictions.
    """
    def __init__(self):
        super(BrierLoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Calculate Brier Score Loss
           
        Args:
            predictions: Predicted probabilities (0-1 range)
            targets: True binary labels (0 or 1)
        
        Returns:
            Brier score (mean squared difference)
        """
        return torch.mean((predictions - targets) ** 2)

class DirectPredictionLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_size=16, num_layers=1, 
                 dropout_rate=0.5, bidirectional=False):
        super(DirectPredictionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Very simple LSTM for tiny datasets
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate the size after LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Minimal classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights specifically for NFL play prediction"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Simple forward pass optimized for small datasets
        Args:
            x: [batch, seq_len, features] - NFL play features
        Returns:
            [batch, 1] - win probabilities
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden_size * directions]
        
        # Use the last output for classification
        final_output = lstm_out[:, -1, :]  # [batch, hidden_size * directions]
        
        # Final classification
        output = self.classifier(final_output)  # [batch, 1]
        
        return output

class DirectLSTMClassifier(BaseDirectClassifier):
    def __init__(self, model, epochs, optimizer, criterion, device, features, scheduler=None, use_scaler=True):
        """
        Direct prediction LSTM classifier
        """
        super().__init__(model, epochs, optimizer, criterion, device, features, scheduler, use_scaler)
    
    def _prepare_data_for_scaling(self, X):
        """Prepare data for scaling - flatten for LSTM"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X.reshape(X.shape[0], -1)
    
    def _apply_scaling(self, X, fit=False):
        """Apply scaling to data - handle 3D input for LSTM"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        original_shape = X.shape
        
        # Flatten for scaling
        X_flattened = X.reshape(X.shape[0], -1)
        
        if self.use_scaler and not self.scaler_fitted and fit:
            X_scaled_flat = self.scaler.fit_transform(X_flattened)
            self.scaler_fitted = True
        elif self.use_scaler and self.scaler_fitted:
            X_scaled_flat = self.scaler.transform(X_flattened)
        else:
            X_scaled_flat = X_flattened
        
        # Reshape back to original shape
        X_scaled = X_scaled_flat.reshape(original_shape)
        return X_scaled
    
    def _get_training_hooks(self):
        """Get training-specific hooks"""
        return {
            'gradient_clip_norm': 0.5,  # Enhanced gradient clipping for LSTM
            'label_smoothing': 0.0,  # Can be adjusted if needed
            'patience': 5  # Aggressive early stopping for tiny datasets
        }
    
    def _get_model_config(self):
        """Extract model configuration for saving"""
        return {
            'input_dim': self.model.input_dim,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'bidirectional': self.model.bidirectional,
        }
    
    def _recreate_model_from_config(self, model_config):
        """Recreate model from saved configuration"""
        return DirectPredictionLSTM(
            input_dim=model_config['input_dim'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            bidirectional=model_config['bidirectional'],
        )
    
    def _get_model_type_name(self):
        """Get model type name"""
        return 'lstm'
    
    def predict(self, x):
        """
        Make predictions on new data
        """
        # Ensure x is numpy array
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Apply scaling if enabled
        if self.use_scaler and self.scaler_fitted:
            # Debug information
            original_shape = x.shape
            
            # Check if shapes are compatible
            expected_features = self.scaler.n_features_in_
            flattened_features = x.shape[0] * np.prod(x.shape[1:]) // x.shape[0] if len(x.shape) > 1 else x.shape[0]
            
            if len(original_shape) >= 2:
                # For LSTM: (batch_size, sequence_length, features) -> (batch_size, sequence_length * features)
                x_flattened = x.reshape(x.shape[0], -1)
                actual_features = x_flattened.shape[1]
                
                if actual_features == expected_features:
                    x_scaled_flat = self.scaler.transform(x_flattened)
                    x_scaled = x_scaled_flat.reshape(original_shape)
                else:
                    print(f"Warning: Scaler feature mismatch. Expected {expected_features}, got {actual_features}")
                    print("Skipping scaling to avoid error. This may affect model performance.")
                    x_scaled = x
            else:
                print("Warning: Unexpected input dimensions. Skipping scaling.")
                x_scaled = x
        else:
            x_scaled = x

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            output = self.model(x_tensor)
            return output.cpu().numpy()
    
    def predict_proba(self, x):
        """
        Return prediction probabilities for each sample.
        Returns array of shape (n_samples, 2) with [prob_class_0, prob_class_1] for each sample.
        """
        preds = self.predict(x)
        
        # Handle single prediction case
        if preds.ndim == 0 or (preds.ndim == 1 and len(preds) == 1):
            pred = preds.item() if preds.ndim > 0 else preds
            return np.array([[1 - pred, pred]])
        
        # Handle multiple predictions case
        # preds should be 1D array of probabilities for positive class
        preds = preds.flatten()  # Ensure 1D
        prob_class_1 = preds
        prob_class_0 = 1 - preds
        
        # Return in sklearn format: [[prob_class_0, prob_class_1], ...]
        return np.column_stack([prob_class_0, prob_class_1])
    
    
    def reset_scaler(self):
        """
        Reset the scaler to handle shape mismatches
        """
        if self.use_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler_fitted = False
            print("Scaler has been reset. It will be re-fitted on next training/prediction.")
    
    def check_scaler_compatibility(self, X):
        """
        Check if the current data is compatible with the fitted scaler
        """
        if not self.use_scaler or not self.scaler_fitted:
            return True
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        expected_features = self.scaler.n_features_in_
        actual_features = X.reshape(X.shape[0], -1).shape[1] if len(X.shape) > 1 else X.shape[0]
        
        return actual_features == expected_features
    
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved DirectLSTMClassifier model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the LSTM model architecture
        model_config = checkpoint['model_config']
        lstm_network = DirectPredictionLSTM(
            input_dim=model_config['input_dim'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            bidirectional=model_config['bidirectional'],
        )
        
        # Load model state
        lstm_network.load_state_dict(checkpoint['model_state_dict'])
        lstm_network.to(device)
        
        # Create classifier instance
        criterion = BrierLoss()
        optimizer = torch.optim.AdamW(lstm_network.parameters(), lr=0.001)
        
        # Get scaler info from checkpoint
        use_scaler = checkpoint.get('use_scaler', True)
        classifier = cls(lstm_network, 1, optimizer, criterion, device, use_scaler=use_scaler)
        
        # Restore scaler if it exists
        if use_scaler and 'scaler' in checkpoint:
            import pickle
            classifier.scaler = pickle.loads(checkpoint['scaler'])
            classifier.scaler_fitted = checkpoint.get('scaler_fitted', False)
        
        # Restore training metrics
        classifier.final_train_loss = checkpoint.get('final_train_loss')
        classifier.final_train_accuracy = checkpoint.get('final_train_accuracy')
        classifier.final_val_loss = checkpoint.get('final_val_loss')
        classifier.final_val_accuracy = checkpoint.get('final_val_accuracy')
        classifier.best_epoch = checkpoint.get('best_epoch')
        classifier.epochs_trained = checkpoint.get('epochs_trained')
        
        print(f"Direct prediction LSTM model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}, Val Loss: {classifier.final_val_loss:.4f}")
        
        return classifier

# Example usage and training script
def setup_direct_lstm_models(training_data, test_data=None, num_models=20, epochs=30, lr=0.001, 
                             batch_size=32, hidden_size=16, num_layers=1, bidirectional=False, use_scaler=True, 
                             save_models=False):
    """
    Setup minimal LSTM models to prevent overfitting on tiny datasets
    Uses very lightweight architecture with aggressive regularization
    
    Args:
        training_data: Dictionary with timesteps as keys and training data as values
        test_data: Optional dictionary with test data. If None, training data is split 90/10
        num_models: Number of models to create across timestep ranges
        epochs: Number of training epochs (default: 30)
        lr: Learning rate (default: 0.001)
        batch_size: Batch size for training (default: 32)
        hidden_size: LSTM hidden dimension size (default: 16)
        num_layers: Number of LSTM layers (default: 1)
        bidirectional: Whether to use bidirectional LSTM (default: False)
        use_scaler: Whether to scale input features (default: True)
        save_models: Whether to save trained models to disk (default: False)
    
    Returns:
        Dictionary of trained minimal LSTM models by timestep range
    """
    models = {}
    for i in range(num_models):
        # Define timestep range
        range_size = 1.0 / (num_models - 1)
        start_time = round(i * range_size, 3)
        end_time = round((i + 1) * range_size, 3)
        timesteps_range = [start_time, end_time]
        
        # Collect data for this timestep range
        X, y = [], []
        X_test, y_test = [], []
        
        for timestep in training_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in training_data[timestep]:
                    X.append(np.array(row['rows'], dtype=np.float32))
                    y.append(np.array(row['label'], dtype=np.float32))
        if test_data:
            for timestep in test_data:
                if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                    for row in test_data[timestep]:
                        X_test.append(np.array(row['rows'], dtype=np.float32))
                        y_test.append(np.array(row['label'], dtype=np.float32))

        if len(X) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
        
        # Handle validation data - either use provided test_data or split training data
        X_full = np.array(X, dtype=np.float32)
        y_full = np.array(y, dtype=np.float32)
        
        if test_data and len(X_test) > 0:
            # Use provided test data as validation
            X_train = X_full
            y_train = y_full
            X_val = np.array(X_test, dtype=np.float32)
            y_val = np.array(y_test, dtype=np.float32)
            print(f"Using provided test data as validation: {len(X_train)} train, {len(X_val)} validation")
        else:
            # Split training data: 90% train, 10% validation
            if len(X_full) < 10:
                # If too few samples, use all for training
                X_train, y_train = X_full, y_full
                X_val, y_val = None, None
                print(f"Too few samples ({len(X_full)}) for splitting, using all for training")
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_full, y_full, test_size=0.1, random_state=42, stratify=y_full
                )
                print(f"Split training data: {len(X_train)} train, {len(X_val)} validation")
        print(f"\nTraining direct prediction LSTM model for timestep range {timesteps_range}")
        
        # Setup simple LSTM model for small datasets
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        lstm_network = DirectPredictionLSTM(
            input_dim=X_train.shape[-1], 
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout_rate=0.5
        )
        criterion = BrierLoss()
        # Simple optimizer with strong regularization
        optimizer = torch.optim.AdamW(
            lstm_network.parameters(), 
            lr=lr, 
            weight_decay=0.1  # Strong weight decay for tiny datasets
        )
        # Simple learning rate scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        
        # Classifier handles scaling internally
        classifier = DirectLSTMClassifier(lstm_network, epochs, optimizer, criterion, device, scheduler, use_scaler=use_scaler)
        
        # Train the model (scaler is handled internally)
        classifier.fit(X_train, y_train, val_X=X_val, val_y=y_val, batch_size=batch_size)
        
        # Save the model
        if save_models:
            model_save_dir = "saved_models_lstm"
            os.makedirs(model_save_dir, exist_ok=True)
            model_prefix = os.path.join(model_save_dir, "nfl_lstm_model")
            saved_filepath = classifier.save_model(model_prefix, timesteps_range)
        
        models[timesteps_range[0]] = classifier
        print(f"NFL LSTM model {i+1}/{num_models} completed")
        
    return models 
