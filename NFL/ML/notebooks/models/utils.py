import numpy as np
def SHAP_analysis(models, timestep, training_data, test_data):
    model = models[timestep]
    X_test = []
    X_train = []
    for row in test_data[timestep]:
        X_test.append(row["rows"].reshape(-1))
    for row in training_data[timestep]:
        X_train.append(row["rows"].reshape(-1)) 
    X_test = np.array(X_test)
    X_train = np.array(X_train)
    # y_test = np.array(y_test)
    vals = model.SHAP_analysis(X_test, X_train, True)
    return vals


def save_SHAP_output(shap_output, name):
    """
    Save the output in a file
    """
    np.savez_compressed(
        (name + ".npz") if name.endswith(".npz") else name,
        values = shap_output.values,
        base_values=shap_output.base_values,
        data=shap_output.data,
        feature_names=np.array(shap_output.feature_names),
    )