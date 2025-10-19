import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
from unittest import mock
from sklearn.base import BaseEstimator
from dvclive import Live

# Import functions from the source files
from src.data_ingestion import split_data
from src.data_processing import processing_data
from src.feature_selection import feature_selection
from src.model_training import model_training
from src.model_evaluation import load_model, save_metrics, evaluate_model

# --- Fixtures for Mock Data and Dependencies ---

@pytest.fixture
def sample_raw_data():
    """A sample DataFrame simulating raw data."""
    return pd.DataFrame({
        'Age': [30, 62, 76, 28, 43, 36],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male'],
        'Blood Type': ['B-', 'A+', 'A-', 'O+', 'AB+', 'A+'],
        'Medical Condition': ['Cancer', 'Obesity', 'Obesity', 'Diabetes', 'Cancer', 'Asthma'],
        'Billing Amount': [18856.28, 33643.33, 27955.10, 37909.78, 14238.32, 48145.11],
        'Room Number': [328, 265, 205, 450, 458, 389],
        'Medication': ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Ibuprofen', 'Penicillin', 'Ibuprofen'],
        'Test Results': ['Normal', 'Inconclusive', 'Normal', 'Abnormal', 'Abnormal', 'Normal']
    })

@pytest.fixture
def sample_processed_data():
    """A sample DataFrame simulating processed data (ready for training/testing)."""
    # This data is derived from the example processed data provided in the prompt
    data = {
        'Gender': [1.0, 0.0, 1.0, 0.0, 1.0],
        'Blood Type': [6.0, 0.0, 0.0, 4.0, 3.0],
        'Medical Condition': [3.0, 3.0, 1.0, 5.0, 5.0],
        'Medication': [0.0, 4.0, 4.0, 1.0, 3.0],
        'Age': [0.28, -0.01, -1.60, 1.15, 0.23],
        'Billing Amount': [-1.56, 0.80, 1.32, 0.13, 0.12],
        'Room Number': [0.33, 0.61, -1.32, -1.43, -0.14],
        'Test Results': [1.0, 0.0, 1.0, 0.0, 1.0] # 1.0=Normal/Abnormal, 0.0=Inconclusive
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    """A mock object that simulates a scikit-learn compatible model."""
    class MockModel(BaseEstimator):
        def fit(self, X, y):
            return self
        def predict(self, X):
            # Predicts '1' for all even-indexed rows and '0' for odd-indexed rows
            return np.array([1 if i % 2 == 0 else 0 for i in range(len(X))])
        def save(self, path):
            pass
    return MockModel()


# --- Tests for Data Ingestion (split_data) ---

def test_split_data_sizes(sample_raw_data):
    """Test if split_data correctly separates train/test data based on test_size."""
    test_size = 0.5
    train_data, test_data = split_data(sample_raw_data, test_size)
    total_size = len(sample_raw_data)
    assert len(train_data) == int(total_size * (1 - test_size))
    assert len(test_data) == int(total_size * test_size)
    assert len(train_data.columns) == len(test_data.columns)


# --- Tests for Data Processing (processing_data) ---

def test_processing_data_output_format(sample_raw_data):
    """Test if processing_data returns a DataFrame with the correct column order and no object types."""
    # Only use a subset of the raw data that doesn't trigger complex OrdinalEncoder issues
    df = processing_data(sample_raw_data)
    expected_last_column = "Test Results"
    # Expected columns after processing (catcol + numcol)
    expected_cols_part = ['Gender', 'Blood Type', 'Medical Condition', 'Medication',
                          'Age', 'Billing Amount', 'Room Number']

    assert isinstance(df, pd.DataFrame)
    # Check if all dtypes are numeric (float/int) except for the last column (Test Results)
    assert not any(df.drop(columns=[expected_last_column]).dtypes == object)
    assert df.columns.tolist()[-1] == expected_last_column
    assert df.columns.tolist()[:-1] == expected_cols_part


# --- Tests for Feature Selection (feature_selection) ---

def test_feature_selection_output_k(sample_processed_data):
    """Test if feature_selection returns a DataFrame with exactly k features + target column."""
    k_features = 3
    df_selected = feature_selection(sample_processed_data, k_features)
    # k_features + 1 (for the target column 'Test Results')
    assert len(df_selected.columns) == k_features + 1
    assert df_selected.columns.tolist()[-1] == 'Test Results'
    assert isinstance(df_selected, pd.DataFrame)


# --- Tests for Model Training (model_training) ---

def test_model_training_returns_estimator(sample_processed_data):
    """Test if model_training returns a trained model object."""
    model = model_training(sample_processed_data)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')


# --- Tests for Model Evaluation (load_model, save_metrics, evaluate_model) ---

@mock.patch('src.model_evaluation.joblib')
def test_load_model_calls_joblib(mock_joblib, mock_model):
    """Test if load_model uses joblib.load with the correct path."""
    mock_joblib.load.return_value = mock_model
    path = "dummy/path/model.pkl"
    model = load_model(path)
    mock_joblib.load.assert_called_once_with(path)
    assert model is mock_model

@mock.patch('src.model_evaluation.os.makedirs')
@mock.patch('src.model_evaluation.open', new_callable=mock.mock_open)
@mock.patch('src.model_evaluation.json.dump')
def test_save_metrics_writes_json(mock_json_dump, mock_open, mock_makedirs):
    """Test if save_metrics creates directory and dumps metrics as JSON."""
    metrics = {
        "accuracy": 0.9,
        "confusion_matrix": np.array([[10, 0], [0, 10]])
    }
    report_dir = "TEST_REPORT"
    save_metrics(metrics, report_dir=report_dir)

    # Check if directory was created
    mock_makedirs.assert_called_once_with(report_dir, exist_ok=True)

    # Check if a file was opened with the correct path
    expected_path = os.path.join(report_dir, "evaluation_metrics.json")
    mock_open.assert_called_once_with(expected_path, 'w')

    # Check if json.dump was called
    mock_json_dump.assert_called_once()
    # Check if numpy array was converted to list before dumping
    dump_args, _ = mock_json_dump.call_args
    dumped_data = dump_args[0]
    assert dumped_data['accuracy'] == 0.9
    assert isinstance(dumped_data['confusion_matrix'], list)

def test_evaluate_model_returns_correct_metrics(mock_model, sample_processed_data):
    """Test if evaluate_model calculates and logs metrics correctly."""
    # Use a small subset of the processed data for predictable results
    test_data = sample_processed_data.head(4)
    mock_live = mock.MagicMock(spec=Live)

    metrics = evaluate_model(mock_model, test_data, mock_live)

    # The mock model predicts: [1, 0, 1, 0]
    # The actual y_test is: [1.0, 0.0, 1.0, 0.0]
    # All 4 predictions are correct (Accuracy = 1.0, Precision=1.0, etc.)

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert pytest.approx(metrics['accuracy'], 0.001) == 1.0

    # Check if dvclive logging was called correctly
    mock_live.log_metric.assert_any_call("accuracy", pytest.approx(1.0))
    mock_live.log_metric.assert_any_call("f1_score", pytest.approx(1.0))
    mock_live.next_step.assert_called_once()