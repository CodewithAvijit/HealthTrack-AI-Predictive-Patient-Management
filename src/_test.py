import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
from unittest import mock
from sklearn.base import BaseEstimator
from dvclive import Live

# Imports are correct (since _test.py is in src/)
from data_ingestion import split_data
from data_processing import processing_data
from feature_selection import feature_selection
from model_training import model_training
from model_evaluation import load_model, save_metrics, evaluate_model

# --- Fixtures for Mock Data and Dependencies ---

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


# --- Tests for Model Training (model_training) ---

def test_model_training_returns_estimator(sample_processed_data):
    """Test if model_training returns a trained model object (verifying training process works)."""
    model = model_training(sample_processed_data)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')


# --- Tests for Model Evaluation (load_model, save_metrics, evaluate_model) ---

# FIX: Change 'src.model_evaluation.joblib' to 'model_evaluation.joblib'
@mock.patch('model_evaluation.joblib')
def test_load_model_calls_joblib(mock_joblib, mock_model):
    """Test if load_model uses joblib.load with the correct path."""
    mock_joblib.load.return_value = mock_model
    path = "dummy/path/model.pkl"
    model = load_model(path)
    mock_joblib.load.assert_called_once_with(path)
    assert model is mock_model

# FIX: Change 'src.model_evaluation.os.makedirs' to 'model_evaluation.os.makedirs'
# FIX: Change 'src.model_evaluation.open' to 'model_evaluation.open'
# FIX: Change 'src.model_evaluation.json.dump' to 'model_evaluation.json.dump'
@mock.patch('model_evaluation.os.makedirs')
@mock.patch('model_evaluation.open', new_callable=mock.mock_open)
@mock.patch('model_evaluation.json.dump')
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

    # Check if json.dump was called
    mock_json_dump.assert_called_once()
    dump_args, _ = mock_json_dump.call_args
    dumped_data = dump_args[0]
    assert dumped_data['accuracy'] == 0.9
    assert isinstance(dumped_data['confusion_matrix'], list)

def test_evaluate_model_performs_prediction_and_eval(mock_model, sample_processed_data):
    """
    Test if evaluate_model calculates metrics correctly and verifies 
    that the model's predict method is called with the expected data (X_test).
    """
    test_data = sample_processed_data.head(4)
    expected_X_test = test_data.iloc[:, :-1]
    
    # Mock the model's predict method for verification
    with mock.patch.object(mock_model, 'predict', wraps=mock_model.predict) as mock_predict:
        
        mock_live = mock.MagicMock(spec=Live)

        metrics = evaluate_model(mock_model, test_data, mock_live)

        # 1. Assert that the model's predict method was called
        mock_predict.assert_called_once()
        
        # 2. Assert that predict was called with the correct X_test data
        pd.testing.assert_frame_equal(mock_predict.call_args[0][0], expected_X_test)

        # 3. Assert the evaluation results are correct for the mock predictions
        assert 'accuracy' in metrics
        assert pytest.approx(metrics['accuracy'], 0.001) == 1.0
        
        # 4. Check dvclive logging
        mock_live.log_metric.assert_any_call("accuracy", pytest.approx(1.0))
        mock_live.next_step.assert_called_once()