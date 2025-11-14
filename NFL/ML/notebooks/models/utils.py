import numpy as np
import os


def SHAP_analysis_timestep(models, timestep, training_data, test_data, plot = True):
    model = models[timestep]
    X_test = []
    X_train = []
    for row in test_data[timestep]:
        X_test.append(row["rows"].reshape(-1))
    for row in training_data[timestep]:
        X_train.append(row["rows"].reshape(-1)) 
    X_test = np.array(X_test)
    X_train = np.array(X_train[:100])
    # y_test = np.array(y_test)
    vals = model.SHAP_analysis(X_test, X_train, plot)
    return vals

def SHAP_analysis(models, training_data, test_data, save_name, save_dir=None):
    """
    Compute SHAP for each timestep and save outputs.
    
    Args:
        models: dict mapping timestep -> model
        training_data: ...
        test_data: ...
        save_name: base name for files (without extension)
        save_dir: optional directory to save into. If None, uses CWD.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for timestep in models.keys():
        shap_output = SHAP_analysis_timestep(models, timestep, training_data, test_data, plot = False)
        filename = save_name + "_" + f"{timestep}"
        save_SHAP_output(shap_output, os.path.join(save_dir, filename))
        print(f"Saved {filename}.npz")


def save_SHAP_output(shap_output, path):
    """
    Save a shap_output into a single .npz file.
    """
    # Build a dict of NPZ-safe arrays
    arrays_to_save = {}
    arrays_to_save["values"] = shap_output.values
    arrays_to_save["base_values"] = shap_output.base_values
    arrays_to_save["data"] = shap_output.data
    arrays_to_save["feature_names"] = np.array(shap_output.feature_names)
    # Save
    np.savez_compressed(
        (path + ".npz") if not path.endswith(".npz") else name,
        **arrays_to_save
    )