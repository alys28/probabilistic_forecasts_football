import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
from .DL_Model import BaseDirectClassifier, NFLDirectDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DirectPredictionTransformer(nn.Module):
    def __init__(self, input_dim=10, d_model=32, nhead=1, num_layers=1, 
                 dim_feedforward=32, dropout_rate=0.2):
        super(DirectPredictionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool across sequence dimension
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights specifically for NFL play prediction"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for transformer-based direct prediction
        Args:
            x: [batch, seq_len, features] or [batch, input_dim] - NFL play features
        Returns:
            [batch, 1] - win probabilities
        """
        batch_size = x.size(0)
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model] for pos encoding
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model] back to batch first
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # Global pooling across sequence dimension
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.global_pool(x)  # [batch, d_model, 1]
        x = x.squeeze(-1)  # [batch, d_model]
        
        # Final classification
        output = self.classifier(x)  # [batch, 1]
        
        return output

class DirectTransformerClassifier(BaseDirectClassifier):
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None, use_scaler=True):
        """
        Direct prediction transformer classifier
        """
        super().__init__(model, epochs, optimizer, criterion, device, scheduler, use_scaler)
    
    def _prepare_data_for_scaling(self, X):
        """Prepare data for scaling - keep original shape for transformer"""
        return X
    
    def _apply_scaling(self, X, fit=False):
        """Apply scaling to data"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.use_scaler and not self.scaler_fitted and fit:
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        elif self.use_scaler and self.scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return X_scaled
    
    def _get_training_hooks(self):
        """Get training-specific hooks"""
        return {
            'gradient_clip_norm': 1.0,  # Gradient clipping for transformer
            'label_smoothing': 0.0,
            'patience': 10
        }
    
    def _get_model_config(self):
        """Extract model configuration for saving"""
        return {
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
            'nhead': self.model.transformer_encoder.layers[0].self_attn.num_heads,
            'num_layers': len(self.model.transformer_encoder.layers),
        }
    
    def _recreate_model_from_config(self, model_config):
        """Recreate model from saved configuration"""
        return DirectPredictionTransformer(
            input_dim=model_config['input_dim'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
        )
    
    def _get_model_type_name(self):
        """Get model type name"""
        return 'transformer'
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=128, verbose=True):
        """
        Train the direct prediction transformer model
        """
        # Prepare and scale data
        X_scaled = self._apply_scaling(X, fit=True)
        val_X_scaled = self._apply_scaling(val_X, fit=False) if val_X is not None else None
        
        # Use base class fit method
        super().fit(X_scaled, y, val_X=val_X_scaled, val_y=val_y, batch_size=batch_size, verbose=verbose)
    
    def predict(self, x):
        """
        Make predictions on new data
        """
        if self.use_scaler and self.scaler_fitted:
            x_scaled = self.scaler.transform(x)
        else:
            x_scaled = x
        # Reshape to (samples, seq_len, input_dim)
        if len(x_scaled.shape) == 2:
            # If 2D, assume it needs to be reshaped
            x_scaled = x_scaled.reshape(x.shape[0], x.shape[1], x.shape[2])

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            output = self.model(x_tensor)
            return output.cpu().numpy()
    
    def predict_proba(self, x):
        """
        Return prediction probabilities (same as predict for this model)
        """
        preds = self.predict(x)
        # Handle single prediction case
        if preds.ndim == 0 or (preds.ndim == 1 and len(preds) == 1):
            pred = preds.item() if preds.ndim > 0 else preds
            return np.array([[1 - pred, pred]])
        # Handle multiple predictions
        preds = preds.flatten()
        return np.column_stack([1 - preds, preds])
    
   
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved DirectTransformerClassifier model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the transformer model architecture
        model_config = checkpoint['model_config']
        transformer_network = DirectPredictionTransformer(
            input_dim=model_config['input_dim'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
        )
        
        # Load model state
        transformer_network.load_state_dict(checkpoint['model_state_dict'])
        transformer_network.to(device)
        
        # Create classifier instance
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(transformer_network.parameters(), lr=0.001)
        
        # Get scaler info from checkpoint
        use_scaler = checkpoint.get('use_scaler', True)
        classifier = cls(transformer_network, 1, optimizer, criterion, device, use_scaler=use_scaler)
        
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
        
        print(f"Direct prediction transformer model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}, Val Loss: {classifier.final_val_loss:.4f}")
        
        return classifier

# Example usage and training script
def setup_direct_transformer_models(training_data, test_data, num_models=20, epochs=100, lr=0.001, 
                                   batch_size=64, d_model=64, nhead=2, num_layers=1, use_scaler=True):
    """
    Setup direct prediction transformer models for each timestep range
    Optimized for NFL play vectors with transformer architecture
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
        
        for timestep in test_data:
            if timestep >= timesteps_range[0] and timestep < timesteps_range[1]:
                for row in test_data[timestep]:
                    X_test.append(np.array(row['rows'], dtype=np.float32))
                    y_test.append(np.array(row['label'], dtype=np.float32))
        
        if len(X) == 0:
            print(f"No data for timestep range {timesteps_range}, skipping...")
            continue
            
        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        print(f"\nTraining direct prediction transformer model for timestep range {timesteps_range}")
        
        # Setup transformer model optimized for NFL play prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        transformer_network = DirectPredictionTransformer(
            input_dim=X_train.shape[-1], 
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 2,
            dropout_rate=0.2,
        )
        criterion = nn.BCELoss()
        # Lower learning rate for transformer
        optimizer = torch.optim.AdamW(transformer_network.parameters(), lr=lr * 0.8, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        
        # Classifier now handles scaling internally (disable scaler for transformer)
        classifier = DirectTransformerClassifier(transformer_network, epochs, optimizer, criterion, device, scheduler, use_scaler=False)
        
        # Train the model (scaler is handled internally)
        classifier.fit(X_train, y_train, val_X=X_test, val_y=y_test, batch_size=batch_size)
        
        # Save the model
        model_save_dir = "saved_models_transformer"
        os.makedirs(model_save_dir, exist_ok=True)
        model_prefix = os.path.join(model_save_dir, "nfl_transformer_model")
        saved_filepath = classifier.save_model(model_prefix, timesteps_range)
        
        models[timesteps_range[0]] = classifier
        print(f"NFL transformer model {i+1}/{num_models} completed")
        
    return models 