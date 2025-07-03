from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np

# User's feature ordering (for reference)
features = ["score_difference", "relative_strength", "type.id", "home_has_possession", 
           "end.down", "end.yardsToEndzone", "end.distance", "field_position_shift", 
           "home_timeouts_left", "away_timeouts_left"]

# Define which features to scale vs. passthrough using column indices
# Features that should be scaled (continuous numeric values)
numeric_feature_indices = [
    0,  # score_difference
    1,  # relative_strength  
    4,  # end.down
    5,  # end.yardsToEndzone
    6,  # end.distance
    7,  # field_position_shift
]

# Features that should NOT be scaled (categorical/binary/discrete)
other_feature_indices = [
    2,  # type.id (categorical)
    3,  # home_has_possession (binary)
    8,  # home_timeouts_left (discrete 0-3)
    9,  # away_timeouts_left (discrete 0-3)
]

# Create named mappings for clarity
numeric_features = [features[i] for i in numeric_feature_indices]
other_features = [features[i] for i in other_feature_indices]

print("Features to be scaled:", numeric_features)
print("Features to passthrough:", other_features)

# Scale the data pipeline (using column indices for numpy arrays)
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_feature_indices),
    ("passthrough", "passthrough", other_feature_indices)
])

# No scaling pipeline (for comparison)
preprocessor_no_scaling = ColumnTransformer(transformers=[
    ("passthrough", "passthrough", list(range(len(features))))
])

# Alternative: If you want to scale ALL features (sometimes useful for neural networks)
preprocessor_scale_all = ColumnTransformer(transformers=[
    ("num", StandardScaler(), list(range(len(features))))
])

# Example usage:
def preprocess_time_series_data(X_train, X_test=None, use_scaling=True):
    """
    Preprocess time series data with shape (n_samples, seq_len, n_features)
    
    Args:
        X_train: Training data (n_samples, seq_len, n_features)
        X_test: Test data (n_samples, seq_len, n_features) [optional]
        use_scaling: Whether to apply feature scaling
    
    Returns:
        Preprocessed X_train, X_test (if provided)
    """
    if use_scaling:
        processor = preprocessor
    else:
        processor = preprocessor_no_scaling
    
    # Reshape to 2D for preprocessing: (n_samples * seq_len, n_features)
    original_train_shape = X_train.shape
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    
    # Fit and transform training data
    X_train_processed = processor.fit_transform(X_train_2d)
    
    # Reshape back to 3D
    X_train_processed = X_train_processed.reshape(original_train_shape[0], original_train_shape[1], -1)
    
    if X_test is not None:
        original_test_shape = X_test.shape
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # Transform test data (don't fit!)
        X_test_processed = processor.transform(X_test_2d)
        X_test_processed = X_test_processed.reshape(original_test_shape[0], original_test_shape[1], -1)
        
        return X_train_processed, X_test_processed
    
    return X_train_processed 