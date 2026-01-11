"""
Pytest configuration and shared fixtures for unit tests.
"""
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture
def sample_zip(tmp_path):
    """
    Create a small test ZIP file with sample content.

    Args:
        tmp_path: pytest fixture providing a temporary directory

    Returns:
        Path: Path to the created ZIP file
    """
    zip_path = tmp_path / "test_dataset.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr("test.txt", "test content")
        zf.writestr("folder/image1.jpg", b"fake image data 1")
        zf.writestr("folder/image2.jpg", b"fake image data 2")
    return zip_path


@pytest.fixture
def corrupted_zip(tmp_path):
    """
    Create a corrupted ZIP file for testing error handling.

    Args:
        tmp_path: pytest fixture providing a temporary directory

    Returns:
        Path: Path to the corrupted ZIP file
    """
    zip_path = tmp_path / "corrupted.zip"
    zip_path.write_text("This is not a valid ZIP file content")
    return zip_path


@pytest.fixture
def sample_directory(tmp_path):
    """
    Create a sample directory structure for testing.

    Args:
        tmp_path: pytest fixture providing a temporary directory

    Returns:
        Path: Path to the sample directory
    """
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()

    # Create subdirectories and files
    (data_dir / "subdir1").mkdir()
    (data_dir / "subdir2").mkdir()

    # Create some test files
    for i in range(7):  # Create 7 files to test the "first 5" logic
        (data_dir / f"file{i}.txt").write_text(f"content {i}")

    (data_dir / "subdir1" / "nested.txt").write_text("nested content")

    return data_dir


@pytest.fixture
def s3_event_single_record():
    """
    Create a sample S3 event with a single record for Lambda testing.

    Returns:
        dict: S3 event structure
    """
    return {
        'Records': [
            {
                's3': {
                    'bucket': {
                        'name': 'test-bucket'
                    },
                    'object': {
                        'key': 'entrada/test-image.jpg'
                    }
                }
            }
        ]
    }


@pytest.fixture
def s3_event_multiple_records():
    """
    Create a sample S3 event with multiple records for Lambda testing.

    Returns:
        dict: S3 event structure with multiple records
    """
    return {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'entrada/image1.jpg'}
                }
            },
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'entrada/image2.jpg'}
                }
            }
        ]
    }


@pytest.fixture
def mock_sagemaker_response_benign():
    """
    Mock SageMaker endpoint response for benign classification.

    Returns:
        dict: Mocked boto3 response with benign prediction (probability < 0.5)
    """
    return {
        'Body': BytesIO(json.dumps([0.3]).encode('utf-8')),
        'ContentType': 'application/json'
    }


@pytest.fixture
def mock_sagemaker_response_malignant():
    """
    Mock SageMaker endpoint response for malignant classification.

    Returns:
        dict: Mocked boto3 response with malignant prediction (probability >= 0.5)
    """
    return {
        'Body': BytesIO(json.dumps([0.75]).encode('utf-8')),
        'ContentType': 'application/json'
    }


@pytest.fixture
def mock_sagemaker_response_boundary():
    """
    Mock SageMaker endpoint response for boundary case (0.5 threshold).

    Returns:
        dict: Mocked boto3 response with exactly 0.5 probability
    """
    return {
        'Body': BytesIO(json.dumps([0.5]).encode('utf-8')),
        'ContentType': 'application/json'
    }


@pytest.fixture
def mock_s3_image_data():
    """
    Mock S3 GetObject response with fake image data.

    Returns:
        dict: Mocked boto3 response with image bytes
    """
    return {
        'Body': BytesIO(b'fake-image-data-for-testing'),
        'ContentType': 'image/jpeg'
    }


@pytest.fixture
def set_endpoint_env(monkeypatch):
    """
    Set the ENDPOINT_NAME environment variable for Lambda testing.

    Args:
        monkeypatch: pytest fixture for safely patching environment

    Returns:
        str: The endpoint name that was set
    """
    endpoint_name = 'test-endpoint'
    monkeypatch.setenv('ENDPOINT_NAME', endpoint_name)
    return endpoint_name


@pytest.fixture
def mock_kaggle_api(mocker):
    """
    Mock the Kaggle API for testing download functions.

    Args:
        mocker: pytest-mock fixture

    Returns:
        MagicMock: Mocked KaggleApi instance
    """
    mock_api = mocker.MagicMock()

    # Mock the KaggleApi class
    mocker.patch('app.src.data_utils.commons.KaggleApi', return_value=mock_api)

    return mock_api
