import os

class Config:
    def __init__(self):
        # Model hyperparameters
        self.input_dim = 10  # Number of input features
        self.output_dim = 1   # Number of output features (e.g., AQI)
        self.hidden_dim = 64   # Hidden layer dimension
        self.num_heads = 8     # Number of attention heads
        self.num_layers = 6    # Number of transformer layers
        self.dropout = 0.1      # Dropout rate
        self.learning_rate = 0.001  # Learning rate
        self.batch_size = 64    # Batch size
        self.num_epochs = 100    # Number of training epochs

        # Paths
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'AQI_INDIA')
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        self.log_path = os.path.join(os.path.dirname(__file__), '..', 'logs')

        # Ensure paths exist
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

config = Config()