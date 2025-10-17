"""
Example usage of Bayesian optimization for LightGBM/XGBoost models.

This example shows how to use the enhanced LightGBM class with Bayesian optimization.
"""

import numpy as np
from xg_boost import LightGBM, setup_xgboost_models, setup_xgboost_models_optimized

def example_usage():
    """Example of how to use Bayesian optimization with the LightGBM model."""
    
    # Example 1: Single model with Bayesian optimization
    print("=== Example 1: Single Model with Bayesian Optimization ===")
    
    # Create some dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 10)
    y_val = np.random.randint(0, 2, 200)
    
    # Train with Bayesian optimization
    model = LightGBM(
        use_calibration=True,
        optimize_hyperparams=True,
        n_trials=20  # Use fewer trials for demo
    )
    
    model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    
    # Make predictions
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    
    print(f"Model accuracy: {model.score(X_val, y_val):.4f}")
    print(f"Best parameters found: {model.best_params}")
    
    # Example 2: Multiple timesteps with optimization
    print("\n=== Example 2: Multiple Timesteps with Optimization ===")
    
    # Create dummy training and validation data dictionaries
    training_data = {}
    validation_data = {}
    
    timesteps = [0.1, 0.5, 0.9]
    for timestep in timesteps:
        # Create dummy data for each timestep
        n_samples = 500
        n_features = 15
        
        X_timestep = np.random.randn(n_samples, n_features)
        y_timestep = np.random.randint(0, 2, n_samples)
        
        # Convert to expected format
        training_data[timestep] = [
            {"label": label, "rows": X_timestep[i:i+1]} 
            for i, label in enumerate(y_timestep)
        ]
        
        # Validation data
        X_val_timestep = np.random.randn(100, n_features)
        y_val_timestep = np.random.randint(0, 2, 100)
        
        validation_data[timestep] = [
            {"label": label, "rows": X_val_timestep[i:i+1]} 
            for i, label in enumerate(y_val_timestep)
        ]
    
    # Train models with Bayesian optimization
    models = setup_xgboost_models_optimized(
        training_data, 
        validation_data, 
        n_trials=10  # Use fewer trials for demo
    )
    
    print(f"Trained {len(models)} models with Bayesian optimization")
    
    # Example 3: Compare with and without optimization
    print("\n=== Example 3: Comparison with/without Optimization ===")
    
    # Without optimization
    model_no_opt = LightGBM(use_calibration=False, optimize_hyperparams=False)
    model_no_opt.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    accuracy_no_opt = model_no_opt.score(X_val, y_val)
    
    # With optimization
    model_with_opt = LightGBM(use_calibration=False, optimize_hyperparams=True, n_trials=10)
    model_with_opt.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    accuracy_with_opt = model_with_opt.score(X_val, y_val)
    
    print(f"Accuracy without optimization: {accuracy_no_opt:.4f}")
    print(f"Accuracy with optimization: {accuracy_with_opt:.4f}")
    print(f"Improvement: {accuracy_with_opt - accuracy_no_opt:.4f}")

if __name__ == "__main__":
    example_usage()
