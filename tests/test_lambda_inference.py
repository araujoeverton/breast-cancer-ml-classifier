"""
Unit tests for app/src/lambda/lambda_function_inference.py

Tests cover:
- Lambda handler event processing
- S3 object retrieval
- SageMaker endpoint invocation
- Classification logic (benign vs malignant)
- Confidence calculation
- Error handling
"""
import json
import importlib
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

# Import lambda module using importlib to avoid 'lambda' keyword conflict
lambda_module = importlib.import_module('app.src.lambda.lambda_function_inference')
lambda_handler = lambda_module.lambda_handler


class TestLambdaHandler:
    """Test suite for lambda_handler function"""

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_benign_classification(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env,
        monkeypatch
    ):
        """Test classification of benign case (probability < 0.5)"""
        # Set ENDPOINT_NAME to the module
        monkeypatch.setattr(lambda_module, 'ENDPOINT_NAME', set_endpoint_env)

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker response with benign probabilities
        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.7, 0.3]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        result = lambda_handler(s3_event_single_record, None)

        # Verify S3 was called correctly
        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='entrada/test-image.jpg'
        )

        # Verify SageMaker was called
        mock_sagemaker.invoke_endpoint.assert_called_once()
        call_kwargs = mock_sagemaker.invoke_endpoint.call_args.kwargs
        assert call_kwargs['EndpointName'] == set_endpoint_env
        assert call_kwargs['ContentType'] == 'application/x-image'

        # Verify response
        assert result['statusCode'] == 200
        assert 'BENIGN' in result['body']

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_malignant_classification(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test classification of malignant case (probability > 0.5)"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker response with malignant probabilities
        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.25, 0.75]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        result = lambda_handler(s3_event_single_record, None)

        # Verify response indicates malignant
        assert result['statusCode'] == 200
        assert 'MALIGNANT' in result['body']

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_boundary_classification(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test classification at boundary (probability = 0.5)"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker response with exactly 0.5 probability
        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.5, 0.5]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        result = lambda_handler(s3_event_single_record, None)

        # At boundary 0.5, should NOT be classified as malignant (> 0.5 required)
        assert result['statusCode'] == 200
        assert 'BENIGN' in result['body']

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_confidence_calculation_benign(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env,
        capsys
    ):
        """Test confidence calculation for benign classification"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker response
        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.85, 0.15]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        lambda_handler(s3_event_single_record, None)

        # Capture output to check confidence value
        captured = capsys.readouterr()

        # For benign diagnosis, confidence should be prob_benign (0.85 = 85%)
        assert 'BENIGN' in captured.out
        assert '85' in captured.out  # Should show 85% confidence

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_confidence_calculation_malignant(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env,
        capsys
    ):
        """Test confidence calculation for malignant classification"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker response
        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.2, 0.8]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        lambda_handler(s3_event_single_record, None)

        # Capture output to check confidence value
        captured = capsys.readouterr()

        # For malignant diagnosis, confidence should be prob_malignant (0.8 = 80%)
        assert 'MALIGNANT' in captured.out
        assert '80' in captured.out  # Should show 80% confidence

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_multiple_records_processing(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_multiple_records,
        mock_s3_image_data,
        set_endpoint_env,
        monkeypatch
    ):
        """Test processing multiple S3 records in a single event"""
        # Set ENDPOINT_NAME to the module
        monkeypatch.setattr(lambda_module, 'ENDPOINT_NAME', set_endpoint_env)

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Create a function that returns a new BytesIO each time it's called
        def mock_invoke(*args, **kwargs):
            return {
                'Body': BytesIO(json.dumps([0.6, 0.4]).encode('utf-8'))
            }

        mock_sagemaker.invoke_endpoint.side_effect = mock_invoke

        # Execute handler
        result = lambda_handler(s3_event_multiple_records, None)

        # Verify S3 was called twice (once per record)
        assert mock_s3.get_object.call_count == 2

        # Verify SageMaker was invoked twice
        assert mock_sagemaker.invoke_endpoint.call_count == 2

        # Verify successful response
        assert result['statusCode'] == 200

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_s3_bucket_and_key_extraction(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test correct extraction of bucket and key from event"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.6, 0.4]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        lambda_handler(s3_event_single_record, None)

        # Verify S3 client was called with correct bucket and key
        mock_s3.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='entrada/test-image.jpg'
        )

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_image_content_passed_to_endpoint(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        set_endpoint_env
    ):
        """Test that image content from S3 is correctly passed to SageMaker"""

        # Setup mocks with specific image data
        test_image_data = b'test-specific-image-bytes'
        mock_s3.get_object.return_value = {
            'Body': BytesIO(test_image_data)
        }

        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.6, 0.4]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        lambda_handler(s3_event_single_record, None)

        # Verify SageMaker endpoint was called with the correct image data
        call_kwargs = mock_sagemaker.invoke_endpoint.call_args.kwargs
        assert call_kwargs['Body'] == test_image_data

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_endpoint_name_from_environment(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        monkeypatch
    ):
        """Test that endpoint name is correctly read from environment variable"""

        # Set custom endpoint name directly on the module
        custom_endpoint = 'custom-test-endpoint'
        monkeypatch.setattr(lambda_module, 'ENDPOINT_NAME', custom_endpoint)

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.6, 0.4]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        lambda_handler(s3_event_single_record, None)

        # Verify custom endpoint name was used
        call_kwargs = mock_sagemaker.invoke_endpoint.call_args.kwargs
        assert call_kwargs['EndpointName'] == custom_endpoint

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_response_format(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test that Lambda response has correct format"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        sagemaker_response = {
            'Body': BytesIO(json.dumps([0.6, 0.4]).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler
        result = lambda_handler(s3_event_single_record, None)

        # Verify response structure
        assert 'statusCode' in result
        assert 'body' in result
        assert result['statusCode'] == 200

        # Verify body is a string (JSON dumped)
        assert isinstance(result['body'], str)

        # Verify body contains diagnosis information
        assert 'Diagnosis' in result['body']
        assert ('BENIGN' in result['body'] or 'MALIGNANT' in result['body'])


class TestLambdaHandlerErrorCases:
    """Test suite for error handling in lambda_handler"""

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_s3_client_error(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        set_endpoint_env
    ):
        """Test handling when S3 client raises an error"""
        from botocore.exceptions import ClientError

        # Mock S3 to raise an error
        mock_s3.get_object.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}},
            'GetObject'
        )

        # Execute handler - should raise exception
        with pytest.raises(ClientError):
            lambda_handler(s3_event_single_record, None)

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_sagemaker_endpoint_error(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test handling when SageMaker endpoint raises an error"""
        from botocore.exceptions import ClientError

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker to raise an error
        mock_sagemaker.invoke_endpoint.side_effect = ClientError(
            {'Error': {'Code': 'ModelError', 'Message': 'Model inference failed'}},
            'InvokeEndpoint'
        )

        # Execute handler - should raise exception
        with pytest.raises(ClientError):
            lambda_handler(s3_event_single_record, None)

    @patch.object(lambda_module, 'sm_runtime')
    @patch.object(lambda_module, 's3_client')
    def test_invalid_sagemaker_response_format(
        self,
        mock_s3,
        mock_sagemaker,
        s3_event_single_record,
        mock_s3_image_data,
        set_endpoint_env
    ):
        """Test handling when SageMaker returns unexpected response format"""

        # Setup mocks
        mock_s3.get_object.return_value = mock_s3_image_data

        # Mock SageMaker with invalid response (single value instead of list)
        sagemaker_response = {
            'Body': BytesIO(json.dumps(0.5).encode('utf-8'))
        }
        mock_sagemaker.invoke_endpoint.return_value = sagemaker_response

        # Execute handler - should raise exception when trying to access indices
        with pytest.raises((IndexError, TypeError)):
            lambda_handler(s3_event_single_record, None)
