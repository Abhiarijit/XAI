import os
import sys
from src.config import Config
from src.data.dataset import AQIDataset
from src.models.transformer import Transformer
from src.training.trainer import Trainer

def main():
    # Load configuration
    config = Config()

    # Initialize dataset
    dataset = AQIDataset(config.data_path)
    
    # Initialize model
    model = Transformer(config)

    # Initialize trainer
    trainer = Trainer(model, dataset, config)

    # Start training or evaluation
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'evaluate':
        trainer.evaluate()
    else:
        print("Invalid mode. Please choose 'train' or 'evaluate'.")
        sys.exit(1)

if __name__ == "__main__":
    main()