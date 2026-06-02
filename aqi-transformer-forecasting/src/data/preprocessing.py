from typing import Any, Dict
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw AQI data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    # Example: Fill missing values with the median of each column
    for column in df.columns:
        if df[column].isnull().any():
            df[column].fillna(df[column].median(), inplace=True)
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract relevant features from the DataFrame."""
    # Example: Create new features based on existing ones
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['day'] = pd.to_datetime(df['date']).dt.dayofweek
    return df

def preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the AQI data."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = extract_features(df)
    return df

def normalize_data(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Normalize the feature columns in the DataFrame."""
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
    return df

def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """Split the DataFrame into training, validation, and test sets."""
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }