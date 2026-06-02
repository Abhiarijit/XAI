import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='True Values', color='green')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.title('Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()