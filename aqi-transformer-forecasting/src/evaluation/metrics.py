from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for model performance.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values from the model.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'R^2 Score': r2
    }