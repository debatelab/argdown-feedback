"""Integration tests for client-server communication."""

from unittest.mock import patch, MagicMock

from argdown_feedback.api.client.builders import create_arganno_request
from argdown_feedback.api.shared.models import VerificationRequest


class TestClientServerIntegration:
    """Test client and server integration."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_request_builder_with_api(self, mock_registry, api_client):
        """Test that request builders work with API endpoints."""
        # Create request using builder
        request = create_arganno_request("Test argument content", "Integration test").build()
        
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = ["TestHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Send request to API
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        # Verify request was processed
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "is_valid" in data
        assert "results" in data
        assert "executed_handlers" in data
        
    def test_client_request_serialization(self, api_client):
        """Test client request serialization and deserialization."""
        # Create complex request - using direct model creation since builders are simpler
        request = VerificationRequest(
            inputs="Complex argument with multiple premises",
            source="Serialization test",
            config={}
        )
        
        # Serialize to JSON
        request_json = request.model_dump()
        
        # Deserialize back to model
        deserialized = VerificationRequest(**request_json)
        
        # Verify data integrity
        assert deserialized.inputs == request.inputs
        assert deserialized.source == request.source
        assert deserialized.config == request.config
        
        # Test with API
        with patch('argdown_feedback.api.server.services.verifier_registry') as mock_registry:
            mock_handler = MagicMock()
            mock_result = MagicMock()
            mock_result.request_id = "serial-test-123"
            mock_result.results = []
            mock_result.executed_handlers = []
            mock_handler.process.return_value = mock_result
            mock_registry.create_handler.return_value = mock_handler
            
            response = api_client.post("/api/v1/verify/infreco", json=request_json)
            assert response.status_code == 200


class TestMultipleVerifierIntegration:
    """Test integration with multiple verifiers."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_multiple_verifier_calls(self, mock_registry, api_client, sample_request_data):
        """Test calling multiple verifiers with same data."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = []
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        verifiers = ["arganno", "argmap", "infreco"]
        responses = []
        
        for verifier in verifiers:
            response = api_client.post(f"/api/v1/verify/{verifier}", json=sample_request_data)
            assert response.status_code == 200
            responses.append(response.json())
            
        # Verify all calls succeeded
        assert len(responses) == 3
        for response in responses:
            assert "verifier" in response
            assert "verifier" in response
            
    def test_verifier_specific_configurations(self, api_client):
        """Test that different verifiers can use different configurations."""
        base_data = {
            "inputs": "Test argument",
            "source": "Config test"
        }
        
        # Different configs for different verifiers
        configs = {
            "arganno": {},
            "argmap": {},
            "infreco": {"from_key": "premises"},
            "logreco": {"from_key": "argument"}
        }
        
        with patch('argdown_feedback.api.server.services.verification_service.verifier_registry') as mock_registry:
            mock_handler = MagicMock()
            mock_result = MagicMock()
            mock_result.results = []
            mock_result.executed_handlers = []
            mock_handler.process.return_value = mock_result
            mock_registry.create_handler.return_value = mock_handler
            
            for verifier, config in configs.items():
                request_data = base_data.copy()
                request_data["config"] = config
                
                response = api_client.post(f"/api/v1/verify/{verifier}", json=request_data)
                assert response.status_code == 200


class TestDataFlowIntegration:
    """Test data flow between client and server components."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_request_processing_pipeline(self, mock_registry, api_client, sample_verification_request):
        """Test complete request processing pipeline."""
        # Setup mock handler with proper response structure
        mock_handler = MagicMock()
        mock_result = MagicMock()
        # Create proper mock results with required fields
        mock_result_item1 = MagicMock()
        mock_result_item1.verifier_id = "handler1"
        mock_result_item1.is_valid = True
        mock_result_item1.message = "Success"
        mock_result_item1.verification_data_references = []
        
        mock_result_item2 = MagicMock()
        mock_result_item2.verifier_id = "handler2"
        mock_result_item2.is_valid = False
        mock_result_item2.message = "Error found"
        mock_result_item2.verification_data_references = []
        
        mock_result.results = [mock_result_item1, mock_result_item2]
        mock_result.executed_handlers = ["ProcessingHandler", "ValidationHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Send request to a real verifier (arganno)
        response = api_client.post("/api/v1/verify/arganno", json=sample_verification_request.model_dump())
        assert response.status_code == 200
        
        # Verify complete data flow
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "is_valid" in data
        assert "results" in data
        assert "executed_handlers" in data
        
    def test_error_propagation(self, api_client, sample_request_data):
        """Test that errors are properly propagated through the system."""
        # Test various error conditions
        error_cases = [
            ("invalid_verifier", 404),
            ("arganno", 422)  # Invalid data will cause validation error
        ]
        
        for verifier, expected_status in error_cases:
            if expected_status == 422:
                # Send invalid data for validation error
                invalid_data = {"invalid": "request"}
                response = api_client.post(f"/api/v1/verify/{verifier}", json=invalid_data)
            else:
                # Send valid data to invalid verifier
                response = api_client.post(f"/api/v1/verify/{verifier}", json=sample_request_data)
                
            assert response.status_code == expected_status
            
            # Verify error response format
            if response.status_code != 200:
                data = response.json()
                assert "detail" in data


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_concurrent_verification_requests(self, mock_registry, api_client, sample_request_data):
        """Test handling multiple concurrent requests."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = []
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Simulate concurrent requests
        num_requests = 5
        responses = []
        
        for i in range(num_requests):
            response = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
            responses.append(response)
            
        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "verifier" in data
            
        # Verify handler was called for each request
        assert mock_handler.process.call_count == num_requests