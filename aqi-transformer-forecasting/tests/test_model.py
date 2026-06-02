import pytest
import torch
from src.models.transformer import Transformer

def test_transformer_initialization():
    model = Transformer(input_dim=10, model_dim=64, num_heads=8, num_layers=6, output_dim=1)
    assert model is not None
    assert isinstance(model, Transformer)

def test_transformer_forward_pass():
    model = Transformer(input_dim=10, model_dim=64, num_heads=8, num_layers=6, output_dim=1)
    sample_input = torch.rand(32, 30, 10)  # (batch_size, sequence_length, input_dim)
    output = model(sample_input)
    assert output.shape == (32, 1)  # (batch_size, output_dim)

def test_transformer_shape_mismatch():
    model = Transformer(input_dim=10, model_dim=64, num_heads=8, num_layers=6, output_dim=1)
    sample_input = torch.rand(32, 30, 5)  # Incorrect input dimension
    with pytest.raises(RuntimeError):
        model(sample_input)

def test_transformer_parameter_count():
    model = Transformer(input_dim=10, model_dim=64, num_heads=8, num_layers=6, output_dim=1)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0  # Ensure the model has parameters

def test_transformer_training_step():
    model = Transformer(input_dim=10, model_dim=64, num_heads=8, num_layers=6, output_dim=1)
    sample_input = torch.rand(32, 30, 10)
    target = torch.rand(32, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    optimizer.zero_grad()
    output = model(sample_input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    
    assert loss.item() < 1.0  # Check if loss is decreasing (arbitrary threshold)