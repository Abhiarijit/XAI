import pytest
import pandas as pd
from src.data.dataset import AQIDataset

def test_aqi_dataset_initialization():
    # Test initialization of the AQIDataset
    dataset = AQIDataset(data_path='data/raw/AQI_INDIA/city_day.csv', sequence_length=30)
    assert dataset is not None
    assert len(dataset) > 0

def test_aqi_dataset_sample():
    # Test retrieval of a sample from the dataset
    dataset = AQIDataset(data_path='data/raw/AQI_INDIA/city_day.csv', sequence_length=30)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2  # (input, target)
    assert sample[0].shape == (30, dataset.num_features)  # Check input shape
    assert sample[1].shape == (dataset.num_features,)  # Check target shape

def test_aqi_dataset_length():
    # Test the length of the dataset
    dataset = AQIDataset(data_path='data/raw/AQI_INDIA/city_day.csv', sequence_length=30)
    assert len(dataset) == dataset.expected_length  # Assuming expected_length is defined in the dataset class

def test_aqi_dataset_invalid_path():
    # Test handling of an invalid data path
    with pytest.raises(FileNotFoundError):
        AQIDataset(data_path='invalid/path.csv', sequence_length=30)