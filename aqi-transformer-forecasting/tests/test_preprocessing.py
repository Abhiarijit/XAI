import pytest
import pandas as pd
from src.data.preprocessing import preprocess_data

def test_preprocess_data():
    # Create a sample DataFrame for testing
    data = {
        'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'aqi': [50, None, 70],
        'feature1': [1.0, 2.0, None],
        'feature2': [None, 3.0, 4.0]
    }
    df = pd.DataFrame(data)

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Check if missing values are handled correctly
    assert processed_df['aqi'].isna().sum() == 0
    assert processed_df['feature1'].isna().sum() == 0
    assert processed_df['feature2'].isna().sum() == 0

    # Check if the date column is in datetime format
    assert pd.api.types.is_datetime64_any_dtype(processed_df['date'])

    # Check if the shape of the DataFrame remains the same
    assert processed_df.shape == df.shape

    # Additional checks can be added based on preprocessing logic
    # For example, checking specific values or ranges in the processed DataFrame
