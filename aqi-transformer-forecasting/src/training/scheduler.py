import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = [base_lr * (1 - progress) for base_lr in self.base_lrs]
        return lr

def get_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int):
    return CustomScheduler(optimizer, warmup_steps, total_steps)