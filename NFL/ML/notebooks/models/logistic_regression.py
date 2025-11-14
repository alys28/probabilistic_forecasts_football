from sklearn.linear_model import LogisticRegression
import numpy as np
from models.Model import Model


class LogisticRegressionModel(Model):
    """Logistic Regression with automatic feature preprocessing"""
    
    def __init__(self, numeric_features, other_features, all_features, use_calibration=False, optimize_hyperparams=False, n_trials=50):
        super().__init__(
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams, 
            numeric_features=numeric_features,
            other_features=other_features,
            all_features=all_features,
            n_trials=n_trials
        )
        
        # Default parameters
        self.params = {
            'max_iter': 2000,
            'random_state': 42,
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs'
        }
    
    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Bayesian optimization."""
        return {
            'C': trial.suggest_float('C', 0.01, 100.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
        }
    
    def _fixed_params(self):
        return {
            'random_state': 42
        }
    
    def _train_model(self, X_train, y_train, X_val, y_val, params):
        """Train logistic regression model and return validation loss."""
        # Merge search params with fixed params
        model_params = {**params, **self._fixed_params()}
        
        # Handle solver-penalty compatibility
        if model_params['penalty'] == 'elasticnet':
            model_params['solver'] = 'saga'
        else:
            model_params['l1_ratio'] = None
        if model_params['penalty'] == 'l1' and model_params['solver'] not in ['liblinear', 'saga']:
            model_params['solver'] = 'liblinear'
        
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        
        # Calculate validation loss (binary cross entropy)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-15) + (1-y_val) * np.log(1-y_val_pred + 1e-15))
        
        return float(val_loss)
    
    def fit(self, X, y, val_X=None, val_y=None):
        """Fit the model with optional hyperparameter optimization."""
        if np.isnan(y).any():
            raise ValueError("NaN values found in target variable")
        
        # Preprocess features
        X_proc = self.fit_transform_X(X)
        
        # Handle validation split
        if val_X is None and val_y is None:
            X_train, X_val, y_train, y_val = self.split_data(X_proc, y, test_size=0.15, random_state=42, stratify=False)
        else:
            X_train, y_train = X_proc, y
            X_val, y_val = self.transform_X(val_X), val_y
        
        # Perform Bayesian optimization if requested
        if self.optimize_hyperparams:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=self.n_trials)
            if best_params['penalty'] == 'elasticnet' and best_params['solver'] != 'saga':
                best_params['solver'] = 'saga'
            elif best_params['penalty'] == 'l1' and best_params['solver'] not in ['liblinear', 'saga']:
                best_params['solver'] = 'liblinear'
            self.params.update(best_params)
        
        # Train final model
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)
        # Fit calibrator if requested
        if self.use_calibration:
            y_val_pred = self.model.predict_proba(X_val)[:, 1]
            self.fit_calibrator(y_val_pred, y_val)
        # Calculate training loss
        y_pred = self.model.predict_proba(X_train)[:, 1]  # Get probability predictions
        train_loss = -np.mean(y_train * np.log(y_pred + 1e-15) + (1-y_pred) * np.log(1-y_pred + 1e-15))  # Binary cross entropy
        train_accuracy = self.model.score(X_train, y_train)

        y_test_pred = self.model.predict_proba(X_val)[:, 1]
        test_loss = -np.mean(y_val * np.log(y_test_pred + 1e-15) + (1-y_val) * np.log(1-y_test_pred + 1e-15))  # Binary cross entropy 
        test_accuracy = self.model.score(X_val, y_val)

        # Print results with optimization info
        opt_info = f" (Optimized)" if self.optimize_hyperparams else ""
        cal_info = f" (Calibrated)" if self.use_calibration else ""
        print(f"{opt_info}{cal_info}: Training Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

    
    def predict(self, X):
        """Make binary predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_proc = self.transform_X(X)
        return self.model.predict(X_proc)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_proc = self.transform_X(X)
        probs = self.model.predict_proba(X_proc)
        
        # Apply calibration if available
        if self.use_calibration and self.calibrator is not None:
            calibrated_probs = self.apply_calibration(probs[:, 1])
            return np.column_stack([1 - calibrated_probs, calibrated_probs])
        
        return probs
    
    


def setup_logistic_regression_models(training_data, test_data, numeric_features, other_features, all_features, use_calibration=False, optimize_hyperparams=False, n_trials=50):
    """
    Setup Logistic Regression models with optional hyperparameter optimization.
    
    Args:
        training_data: Dictionary with timestep keys and training data
        test_data: Dictionary with timestep keys and test data
        numeric_features: List of numeric feature names
        other_features: List of non-numeric feature names
        all_features: List of all feature names in order
        use_calibration: Whether to use probability calibration
        optimize_hyperparams: Whether to perform Bayesian optimization
        n_trials: Number of optimization trials (only used if optimize_hyperparams=True)
    """
    models = {}
    for timestep in training_data:
        print(f"Processing timestep: {timestep}")
        X = training_data[timestep]
        y = np.array([row["label"] for row in X])
        # Check for NaN in y
        if np.isnan(y).any():
            print(f"NaN found in y for timestep: {timestep}")
            continue
        X = np.array([row["rows"].reshape(-1) for row in X])
        
        # Prepare test data
        X_test = None
        y_test = None
        if test_data:
            y_test = np.array([row["label"] for row in test_data[timestep]])
            X_test = np.array([row["rows"].reshape(-1) for row in test_data[timestep]])
        
        # Create and train model
        model = LogisticRegressionModel(
            numeric_features=numeric_features,
            other_features=other_features,
            all_features=all_features,
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials
        )
        print(f"Timestep {timestep:.2%}", end=" ")
        model.fit(X, y, val_X=X_test, val_y=y_test)
        
        models[timestep] = model
    return models