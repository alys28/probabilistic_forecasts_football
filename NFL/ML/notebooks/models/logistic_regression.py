from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionModel:
    """Logistic Regression with automatic feature preprocessing"""
    
    def __init__(self):
        # Features that should be scaled (continuous/numeric values)
        self.numeric_features = [
            "score_difference",      # Score differential
            "relative_strength",     # Team strength difference  
            "end.yardsToEndzone",   # Distance to end zone
            "end.distance",         # Distance to first down
            "field_position_shift", # Change in field position
        ]
        
        # Features that should NOT be scaled (categorical/binary/discrete)
        self.other_features = [
            "timestep",             # Game completion percentage
            "type.id",             # Play type (categorical)
            "home_has_possession", # Binary indicator
            "end.down",            # Down number (1-4, discrete)
            "home_timeouts_left",  # Discrete count (0-3)
            "away_timeouts_left",  # Discrete count (0-3)
        ]
        
        # Standard feature order
        self.all_features = [
            "score_difference", "timestep", "type.id", "relative_strength", 
            "home_has_possession", "end.down", "end.yardsToEndzone", "end.distance", 
            "field_position_shift", "home_timeouts_left", "away_timeouts_left"
        ]
        
        # Generate feature indices
        self.numeric_indices = [self.all_features.index(f) for f in self.numeric_features if f in self.all_features]
        self.other_indices = [self.all_features.index(f) for f in self.other_features if f in self.all_features]
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        """Create the preprocessing + model pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.numeric_indices),
                ("other", "passthrough", self.other_indices)
            ],
            remainder="passthrough"
        )
        
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ])
    
    def fit(self, X, y):
        """Fit the model"""
        if np.isnan(y).any():
            raise ValueError("NaN values found in target variable")
        return self.pipeline.fit(X, y)
    
    def predict(self, X):
        """Make binary predictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions"""
        return self.pipeline.predict_proba(X)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        return self.pipeline.score(X, y)

def setup_logistic_regression_models(training_data, test_data=None):
    """Set up logistic regression models for each timestep"""
    models = {}
    
    for timestep in training_data:
        print(f"Processing timestep: {timestep}")
        
        # Prepare training data
        X_train_data = training_data[timestep]
        y_train = np.array([row["label"] for row in X_train_data])
        X_train = np.array([row["rows"].reshape(-1) for row in X_train_data])
        
        # Check for NaN in labels
        if np.isnan(y_train).any():
            print(f"NaN found in labels for timestep: {timestep}, skipping...")
            continue
        
        # Train model
        model = LogisticRegressionModel()
        model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = model.predict_proba(X_train)[:, 1]
        train_loss = -np.mean(y_train * np.log(y_train_pred + 1e-15) + (1-y_train) * np.log(1-y_train_pred + 1e-15))
        train_accuracy = model.score(X_train, y_train)
        
        # Calculate test metrics if available
        if test_data:
            X_test_data = test_data[timestep]
            y_test = np.array([row["label"] for row in X_test_data])
            X_test = np.array([row["rows"].reshape(-1) for row in X_test_data])
            
            y_test_pred = model.predict_proba(X_test)[:, 1]
            test_loss = -np.mean(y_test * np.log(y_test_pred + 1e-15) + (1-y_test) * np.log(1-y_test_pred + 1e-15))
            test_accuracy = model.score(X_test, y_test)
            
            print(f"  Timestep {timestep:.2%}: Train Loss={train_loss:.4f}, Acc={train_accuracy:.4f} | Test Loss={test_loss:.4f}, Acc={test_accuracy:.4f}")
        else:
            print(f"  Timestep {timestep:.2%}: Train Loss={train_loss:.4f}, Acc={train_accuracy:.4f}")
        
        models[timestep] = model
    
    return models