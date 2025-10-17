import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

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

class DirectPredictionNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, dropout_rate=0.2):
        super(DirectPredictionNetwork, self).__init__()
        
        # Optimized architecture for NFL play vectors
        # Simple network without batch normalization to avoid single-batch issues
        self.network = nn.Sequential(
            # First layer: expand from input_dim to hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer: compress to smaller representation
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout in later layers
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout in later layers
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout in later layers
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout in later layers
            
            # Output layer - single neuron for binary classification
            nn.Linear(hidden_dim // 16, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
        # Initialize weights for better convergence with small input
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights specifically for NFL play prediction"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for direct prediction
        Args:
            x: [batch, n] - NFL play features (n-entry vector)
        Returns:
            [batch, 1] - win probabilities
        """
        return self.network(x)

class DirectClassifier:
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None, use_scaler=True):
        """
        Direct prediction classifier
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
        Train the direct prediction model
        """
        # Apply scaling if enabled (data is already 2D from setup function)
        if self.use_scaler and not self.scaler_fitted:
            print("Fitting scaler on training data...")
            print(f"Training data shape: {X.shape}")
            X_scaled = self.scaler.fit_transform(X)
            print(f"Scaler fitted with {self.scaler.n_features_in_} features")
            self.scaler_fitted = True
        elif self.use_scaler and self.scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        train_dataset = NFLDirectDataset(X_scaled, y, device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_X is not None and val_y is not None:
            # Apply scaling to validation data (already 2D)
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
        
        print(f"Starting training on device: {self.device}")
        
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
                
                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
                # print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
                # 
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
                            print(f"Restored model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
                        break
            else:
                # print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy:.4f}")
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
        # Ensure input is flattened to 2D if it's 3D
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
            
        # Apply scaling if enabled (data should now be 2D)
        if self.use_scaler and self.scaler_fitted:
            x_scaled = self.scaler.transform(x)
        else:
            x_scaled = x
            
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            output = self.model(x_tensor)
            return output.cpu().numpy()
    
    def predict_proba(self, x):
        """
        Return prediction probabilities (same as predict for this model)
        """
        pred = self.predict(x).flatten()
        # Return in sklearn format: [[prob_class_0, prob_class_1], ...]
        return np.column_stack([1 - pred, pred])
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            float: Mean accuracy score
        """
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
        Save the trained model
        """
        val_acc = self.final_val_accuracy if self.final_val_accuracy else 0
        val_loss = self.final_val_loss if self.final_val_loss else 0
        
        filename = f"{filepath_prefix}_direct_{timesteps_range[0]}-{timesteps_range[1]}_ep{self.best_epoch}"
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
                'input_dim': list(self.model.network.children())[0].in_features,
                'hidden_dim': list(self.model.network.children())[0].out_features
            },
            'use_scaler': self.use_scaler,
            'scaler_fitted': self.scaler_fitted
        }
        
        # Save scaler if it exists and is fitted
        if self.use_scaler and self.scaler_fitted:
            import pickle
            checkpoint['scaler'] = pickle.dumps(self.scaler)
        
        torch.save(checkpoint, filename)
        print(f"Direct prediction model saved: {filename}")
        return filename
    
    @classmethod
    def load_model(model_class, filepath, device=None):
        """
        Load a saved DirectClassifier model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate the model architecture
        model_config = checkpoint['model_config']
        direct_network = DirectPredictionNetwork(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim']
        )
        
        # Load model state
        direct_network.load_state_dict(checkpoint['model_state_dict'])
        direct_network.to(device)
        
        # Create classifier instance
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(direct_network.parameters(), lr=0.001)
        
        # Get scaler info from checkpoint
        use_scaler = checkpoint.get('use_scaler', True)
        classifier = model_class(direct_network, 1, optimizer, criterion, device, use_scaler=use_scaler)
        
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
        
        print(f"Direct prediction model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}, Val Loss: {classifier.final_val_loss:.4f}")
        
        return classifier

# Example usage and training script
def setup_direct_models(training_data, test_data=None, num_models=20, epochs=100, lr=0.001, 
                       batch_size=64, hidden_dim=128, use_scaler=True):
    """
    Setup direct prediction models for each timestep range
    Optimized for NFL play vectors
    
    Args:
        training_data: Dictionary with timesteps as keys and training data as values
        test_data: Optional dictionary with test data. If None, training data is split 90/10
        num_models: Number of models to create across timestep ranges
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        hidden_dim: Hidden dimension size for neural network
        use_scaler: Whether to scale input features
    
    Returns:
        Dictionary of trained neural network models by timestep range
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
        
        # Flatten the input data for neural network (convert from 3D to 2D)
        print(f"Original data shape: {X_full.shape}")
        X_full_flattened = X_full.reshape(X_full.shape[0], -1)
        print(f"Flattened data shape: {X_full_flattened.shape}")
        
        if test_data and len(X_test) > 0:
            # Use provided test data as validation
            X_train = X_full_flattened
            y_train = y_full
            X_test_array = np.array(X_test, dtype=np.float32)
            X_val = X_test_array.reshape(X_test_array.shape[0], -1)
            y_val = np.array(y_test, dtype=np.float32)
            print(f"Using provided test data as validation: {len(X_train)} train, {len(X_val)} validation")
        else:
            # Split training data: 90% train, 10% validation
            if len(X_full_flattened) < 10:
                # If too few samples, use all for training
                X_train, y_train = X_full_flattened, y_full
                X_val, y_val = None, None
                print(f"Too few samples ({len(X_full_flattened)}) for splitting, using all for training")
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_full_flattened, y_full, test_size=0.1, random_state=42, stratify=y_full
                )
                print(f"Split training data: {len(X_train)} train, {len(X_val)} validation")
        
        print(f"\nTraining direct prediction model for timestep range {timesteps_range}")
        
        # Setup model optimized for NFL play prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        direct_network = DirectPredictionNetwork(
            input_dim=X_train.shape[1],  # Now correctly using flattened dimension
            hidden_dim=hidden_dim,
            dropout_rate=0.2  # Lower dropout for smaller network
        )
        criterion = nn.MSELoss()
        # Slightly higher learning rate for smaller network
        optimizer = torch.optim.AdamW(direct_network.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5  # More aggressive scheduling
        )
        
        # Classifier now handles scaling internally
        classifier = DirectClassifier(direct_network, epochs, optimizer, criterion, device, scheduler, use_scaler=use_scaler)
        
        # Train the model (scaler is handled internally)
        classifier.fit(X_train, y_train, val_X=X_val, val_y=y_val, batch_size=batch_size)
        
        # Save the model
        model_save_dir = "saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_prefix = os.path.join(model_save_dir, "nfl_direct_model")
        saved_filepath = classifier.save_model(model_prefix, timesteps_range)
        
        models[timesteps_range[0]] = classifier
        print(f"NFL direct model {i+1}/{num_models} completed")
        
    return models