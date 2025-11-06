import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math

class NFLDirectDataset(Dataset):
    def __init__(self, data_x, data_y, device='cpu'):
        """
        Direct prediction dataset - no pairs needed
        Args:
            data_x: Game features
            data_y: Game outcomes (score differences)
        """
        self.device = device
        self.data_x = torch.FloatTensor(data_x).to(device)
        
        # Convert score differences to win/loss labels (1 = win, 0 = loss)
        self.data_y = torch.FloatTensor([(1.0 if y > 0 else 0.0) for y in data_y]).to(device)
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

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

class DirectTransformerClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None, use_scaler=True):
        """
        Direct prediction transformer classifier
        """
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.use_scaler = use_scaler
        
        # Initialize scaler if needed
        if self.use_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler_fitted = False
        else:
            self.scaler = None
            self.scaler_fitted = False
        
        # Track training metrics
        self.final_train_loss = None
        self.final_train_accuracy = None
        self.final_val_loss = None
        self.final_val_accuracy = None
        self.best_epoch = None
        self.epochs_trained = None
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=128):
        """
        Train the direct prediction transformer model
        """
        # Ensure X is 2D for scaler (flatten if needed)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
      
        # Apply scaling if enabled
        if self.use_scaler and not self.scaler_fitted:
            print("Fitting scaler on training data...")
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        elif self.use_scaler and self.scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        train_dataset = NFLDirectDataset(X_scaled, y, device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_X is not None and val_y is not None:
            
            # Apply scaling to validation data
            if self.use_scaler:
                val_X_scaled = self.scaler.transform(val_X)
            else:
                val_X_scaled = val_X
            val_dataset = NFLDirectDataset(val_X_scaled, val_y, device=self.device)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 10
        best_epoch = 0
        
        print(f"Starting transformer training on device: {self.device}")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x).squeeze(-1)  # Remove only the last dimension
                    
                self.optimizer.zero_grad()
                loss = self.criterion(output, y)
                
                loss.backward()
                
                # Gradient clipping for transformer
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = (output > 0.5).float()
                    train_correct += (predictions == y).float().sum().item()
                    train_total += len(predictions)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation
            if val_loader is not None:
                val_loss, val_accuracy = self._evaluate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch + 1
                    # Store best metrics
                    self.final_val_loss = val_loss
                    self.final_val_accuracy = val_accuracy
                    self.final_train_loss = avg_train_loss
                    self.final_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        print(f"Best epoch: {best_epoch}, Train Acc: {self.final_train_accuracy:.4f}, Train Loss: {self.final_train_loss:.4f}, Val Acc: {self.final_val_accuracy:.4f}, Val Loss: {self.final_val_loss:.4f}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            print(f"Restored transformer model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
                        break
            else:
                # Store final metrics for no-validation case
                self.final_train_loss = avg_train_loss
                self.final_train_accuracy = train_accuracy
                best_epoch = epoch + 1
                if best_model_state is None:
                    best_model_state = self.model.state_dict().copy()
        
        # Store final training info
        self.best_epoch = best_epoch
        self.epochs_trained = epoch + 1
    
    def _evaluate(self, data_loader):
        """Helper method for evaluation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Skip batch if inputs contain NaN
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                    
                output = self.model(x).squeeze(-1)
                
                # Skip batch if outputs contain NaN
                if torch.isnan(output).any():
                    continue
                    
                loss = self.criterion(output, y)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (output > 0.5).float()
                correct += (predictions == y).float().sum().item()
                total_samples += len(predictions)
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, x):
        """
        Make predictions on new data
        """
        if self.use_scaler and self.scaler_fitted:
            x_scaled = self.scaler.transform(x)
        else:
            x_scaled = x
        # Reshape to (samples, seq_len, input_dim)
        print(x.shape)
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
        pred = self.predict(x).item()
        return [[1 - pred, pred]]
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            float: Mean accuracy score
        """
        # Ensure X is 2D (flattening is handled in predict method)
        # Get predictions (scaling is handled internally in predict method)
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(float).flatten()
        
        # Convert y to binary if needed (in case it's score differences)
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        y_true = np.array([(1.0 if label > 0 else 0.0) for label in y]) if y.max() > 1 or y.min() < 0 else y
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def get_scaler(self):
        """
        Return the fitted scaler if available
        """
        if self.use_scaler and self.scaler_fitted:
            return self.scaler
        else:
            return None
    
    def save_model(self, filepath_prefix, timesteps_range):
        """
        Save the trained transformer model
        """
        val_acc = self.final_val_accuracy if self.final_val_accuracy else 0
        val_loss = self.final_val_loss if self.final_val_loss else 0
        
        filename = f"{filepath_prefix}_transformer_{timesteps_range[0]}-{timesteps_range[1]}_ep{self.best_epoch}"
        filename += f"_valAcc{val_acc:.4f}_valLoss{val_loss:.4f}.pth"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps_range': timesteps_range,
            'best_epoch': self.best_epoch,
            'epochs_trained': self.epochs_trained,
            'final_train_loss': self.final_train_loss,
            'final_train_accuracy': self.final_train_accuracy,
            'final_val_loss': self.final_val_loss,
            'final_val_accuracy': self.final_val_accuracy,
            'model_config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model,
                'nhead': self.model.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(self.model.transformer_encoder.layers),
            },
            'use_scaler': self.use_scaler,
            'scaler_fitted': self.scaler_fitted
        }
        
        # Save scaler if it exists and is fitted
        if self.use_scaler and self.scaler_fitted:
            import pickle
            checkpoint['scaler'] = pickle.dumps(self.scaler)
        
        torch.save(checkpoint, filename)
        print(f"Direct prediction transformer model saved: {filename}")
        return filename
    
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