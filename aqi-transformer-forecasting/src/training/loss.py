import torch
import torch.nn as nn

class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, predictions, targets):
        return nn.functional.mse_loss(predictions, targets)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = MeanSquaredError()

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        # Add any additional loss components here if needed
        return mse_loss