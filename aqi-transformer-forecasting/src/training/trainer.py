import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import AQIDataset
from src.models.transformer import Transformer
from src.training.loss import compute_loss
from src.utils.logger import Logger

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.logger = Logger()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.logger.log(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
            self.scheduler.step()

            self.validate()

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = compute_loss(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.log(f'Validation Loss: {avg_loss:.4f}')