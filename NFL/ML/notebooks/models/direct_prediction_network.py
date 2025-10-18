import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import optuna
import logging

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
    def __init__(self, input_dim=10, hidden_dims=None, dropout_rate=0.2, num_layers=None):
        super(DirectPredictionNetwork, self).__init__()
        
        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [64, 32, 16, 8]
        if num_layers is None:
            num_layers = len(hidden_dims)
        
        # Ensure we don't exceed the number of provided hidden dimensions
        num_layers = min(num_layers, len(hidden_dims))
        
        # Build dynamic network architecture
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            layers.append(nn.ReLU(inplace=True))
            
            # Add dropout (reduce dropout in later layers)
            dropout_factor = 1.0 if i == 0 else 0.5
            layers.append(nn.Dropout(dropout_rate * dropout_factor))
            
            prev_dim = hidden_dims[i]
        
        # Output layer - single neuron for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability between 0 and 1
        
        self.network = nn.Sequential(*layers)
        
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
    def __init__(self, model, epochs, optimizer, criterion, device, scheduler=None, use_scaler=True,
                 optimize_hyperparams=False, n_trials=30, optimization_epochs=None):
        """
        Direct prediction classifier with optional hyperparameter optimization
        
        Args:
            model: DirectPredictionNetwork model
            epochs: Number of training epochs
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: PyTorch device
            scheduler: Learning rate scheduler (optional)
            use_scaler: Whether to use feature scaling
            optimize_hyperparams: Whether to run Optuna optimization
            n_trials: Number of Optuna trials (if optimization enabled)
            optimization_epochs: Epochs per trial (if None, uses epochs//2)
        """
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.use_scaler = use_scaler
        
        # Optimization parameters
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.optimization_epochs = optimization_epochs if optimization_epochs else max(epochs // 2, 10)
        
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
        
        # Optimization results
        self.optimization_results = None
        self.best_hyperparams = None
        self.optuna_study = None
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=128, verbose = True):
        """
        Train the direct prediction model with optional hyperparameter optimization
        """
        # Apply scaling if enabled (data is already 2D from setup function)
        if self.use_scaler and not self.scaler_fitted:
            # print("Fitting scaler on training data...")
            # print(f"Training data shape: {X.shape}")
            X_scaled = self.scaler.fit_transform(X)
            # print(f"Scaler fitted with {self.scaler.n_features_in_} features")
            self.scaler_fitted = True
        elif self.use_scaler and self.scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Run hyperparameter optimization if enabled
        if self.optimize_hyperparams and val_X is not None and val_y is not None:
            print("Running hyperparameter optimization...")
            best_params, best_value = self._run_hyperparameter_optimization(X_scaled, y, val_X, val_y)
            
            # Recreate model with best parameters
            hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(best_params['num_layers'])]
            # Create new model with optimized architecture
            self.model = DirectPredictionNetwork(
                input_dim=X_scaled.shape[1],
                hidden_dims=hidden_dims,
                dropout_rate=best_params['dropout_rate'],
                num_layers=best_params['num_layers']
            ).to(self.device)
            
            # Update optimizer with best parameters
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=best_params['learning_rate'],
                weight_decay=best_params['weight_decay']
            )
            
            # Update scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=3
            )
            
            # Update batch size
            batch_size = best_params['batch_size']
            if verbose:
                print(f"Model recreated with optimized parameters. Best validation accuracy: {best_value:.4f}")
        
        # Continue with normal training
        
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
        
        # print(f"Starting training on device: {self.device}")
        
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
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                            print(f"Best epoch: {best_epoch}, Train Acc: {self.final_train_accuracy:.4f}, Train Loss: {self.final_train_loss:.4f}, Val Acc: {self.final_val_accuracy:.4f}, Val Loss: {self.final_val_loss:.4f}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            if verbose:
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
    
    def _optuna_objective(self, trial, X_train, y_train, X_val, y_val, input_dim):
        """
        Optuna objective function for hyperparameter optimization
        """
        # Define hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        num_layers = trial.suggest_int('num_layers', 2, 6)
        
        # Define hidden layer dimensions
        hidden_dims = []
        for i in range(num_layers):
            dim = trial.suggest_int(f'hidden_dim_{i}', 16, 256)
            hidden_dims.append(dim)
        
        # Ensure decreasing architecture (optional constraint)
        if trial.suggest_categorical('enforce_decreasing', [True, False]):
            hidden_dims = sorted(hidden_dims, reverse=True)
        
        # Weight decay
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Create model with suggested hyperparameters
        model = DirectPredictionNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
        
        # Setup optimizer and criterion
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=3
        )
        
        # Create temporary classifier for this trial
        temp_classifier = DirectClassifier(model, self.optimization_epochs, optimizer, criterion, 
                                        self.device, scheduler, use_scaler=True)
        
        # Train the model
        temp_classifier.fit(X_train, y_train, val_X=X_val, val_y=y_val, batch_size=batch_size, verbose=False)
        
        # Return validation accuracy (Optuna maximizes by default)
        val_loss = temp_classifier.final_val_loss if temp_classifier.final_val_loss else 0.0
        
        # Report intermediate results for pruning
        trial.report(val_loss, step=self.optimization_epochs)
        
        # Handle pruning based on intermediate results
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return val_loss
    
    def _run_hyperparameter_optimization(self, X, y, val_X, val_y):
        """
        Run Optuna hyperparameter optimization
        """
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Configure Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"nfl_direct_optuna_{id(self)}",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y, val_X, val_y, X.shape[1]),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"Best validation accuracy: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Store optimization results
        self.optimization_results = {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }
        self.best_hyperparams = best_params
        self.optuna_study = study
        
        return best_params, best_value
    
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
        pred = self.predict_proba(x)[0][1]
    
    def predict_proba(self, x):
        """
        Return prediction probabilities (same as predict for this model)
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
            pred = output.cpu().numpy().flatten()
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
        y_pred_proba = self.predict_proba(X)[:,1]
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
    
    def get_optimization_results(self):
        """
        Return optimization results if available
        """
        return self.optimization_results
    
    def get_best_hyperparams(self):
        """
        Return best hyperparameters if optimization was run
        """
        return self.best_hyperparams
    
    def get_optuna_study(self):
        """
        Return Optuna study object if optimization was run
        """
        return self.optuna_study
    
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
                'hidden_dims': [layer.out_features for layer in self.model.network.children() 
                               if isinstance(layer, nn.Linear)][:-1],  # Exclude output layer
                'dropout_rate': self.model.network[2].p,  # Get dropout rate from first dropout layer
                'num_layers': len([layer for layer in self.model.network.children() 
                                 if isinstance(layer, nn.Linear)]) - 1  # Exclude output layer
            },
            'use_scaler': self.use_scaler,
            'scaler_fitted': self.scaler_fitted,
            'optimize_hyperparams': self.optimize_hyperparams,
            'n_trials': self.n_trials,
            'optimization_epochs': self.optimization_epochs
        }
        
        # Save scaler if it exists and is fitted
        if self.use_scaler and self.scaler_fitted:
            import pickle
            checkpoint['scaler'] = pickle.dumps(self.scaler)
        
        # Save optimization results if available
        if self.optimization_results is not None:
            checkpoint['optimization_results'] = self.optimization_results
            checkpoint['best_hyperparams'] = self.best_hyperparams
        
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
            hidden_dims=model_config.get('hidden_dims', [64, 32, 16, 8]),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            num_layers=model_config.get('num_layers', 4)
        )
        
        # Load model state
        direct_network.load_state_dict(checkpoint['model_state_dict'])
        direct_network.to(device)
        
        # Create classifier instance
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(direct_network.parameters(), lr=0.001)
        
        # Get scaler info from checkpoint
        use_scaler = checkpoint.get('use_scaler', True)
        optimize_hyperparams = checkpoint.get('optimize_hyperparams', False)
        n_trials = checkpoint.get('n_trials', 30)
        optimization_epochs = checkpoint.get('optimization_epochs', None)
        
        classifier = model_class(direct_network, 1, optimizer, criterion, device, 
                               use_scaler=use_scaler, optimize_hyperparams=optimize_hyperparams,
                               n_trials=n_trials, optimization_epochs=optimization_epochs)
        
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
        
        # Restore optimization results if available
        if 'optimization_results' in checkpoint:
            classifier.optimization_results = checkpoint['optimization_results']
            classifier.best_hyperparams = checkpoint.get('best_hyperparams')
        
        print(f"Direct prediction model loaded from: {filepath}")
        print(f"Best epoch: {classifier.best_epoch}, Val Acc: {classifier.final_val_accuracy:.4f}, Val Loss: {classifier.final_val_loss:.4f}")
        
        return classifier


# Example usage and training script
def setup_direct_models(training_data, test_data=None, num_models=20, epochs=100, lr=0.001, 
                       batch_size=64, hidden_dim=128, use_scaler=True, save_model=False,
                       optimize_hyperparams=False, n_trials=30):
    """
    Setup direct prediction models for each timestep range
    Optimized for NFL play vectors with optional hyperparameter optimization
    
    Args:
        training_data: Dictionary with timesteps as keys and training data as values
        test_data: Optional dictionary with test data. If None, training data is split 90/10
        num_models: Number of models to create across timestep ranges
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        hidden_dim: Hidden dimension size for neural network
        use_scaler: Whether to scale input features
        save_model: Whether to save trained models
        optimize_hyperparams: Whether to run Optuna hyperparameter optimization
        n_trials: Number of Optuna trials (if optimization enabled)
    
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
            hidden_dims=[hidden_dim, hidden_dim//2, hidden_dim//4, hidden_dim//8],
            dropout_rate=0.2,  # Lower dropout for smaller network
            num_layers=4
        )
        criterion = nn.MSELoss()
        # Slightly higher learning rate for smaller network
        optimizer = torch.optim.AdamW(direct_network.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5  # More aggressive scheduling
        )
        
        # Classifier now handles scaling and optimization internally
        classifier = DirectClassifier(direct_network, epochs, optimizer, criterion, device, scheduler, 
                                    use_scaler=use_scaler, optimize_hyperparams=optimize_hyperparams, 
                                    n_trials=n_trials)
        
        # Train the model (scaler is handled internally)
        classifier.fit(X_train, y_train, val_X=X_val, val_y=y_val, batch_size=batch_size)
        if save_model: 
            # Save the model
            model_save_dir = "saved_models"
            os.makedirs(model_save_dir, exist_ok=True)
            model_prefix = os.path.join(model_save_dir, "nfl_direct_model")
            saved_filepath = classifier.save_model(model_prefix, timesteps_range)
        
        models[timesteps_range[0]] = classifier
        print(f"NFL direct model {i+1}/{num_models} completed")
        
    return models