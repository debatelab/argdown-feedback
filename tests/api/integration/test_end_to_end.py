"""End-to-end integration tests for complete workflows."""

from unittest.mock import patch, MagicMock

from argdown_feedback.api.client.builders import create_arganno_request


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_simple_arganno_workflow(self, mock_registry, api_client, sample_request_data):
        """Test complete arganno workflow using fixtures."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = ["ArgannoHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Send request to API
        response = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response.status_code == 200
        
        # Verify workflow completion
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "is_valid" in data
        assert "results" in data
        
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_simple_infreco_workflow(self, mock_registry, api_client, sample_infreco_request_data):
        """Test complete infreco workflow using fixtures."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = ["InfrecoHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Send request to API
        response = api_client.post("/api/v1/verify/infreco", json=sample_infreco_request_data)
        assert response.status_code == 200
        
        # Verify workflow completion
        data = response.json()
        assert data["verifier"] == "infreco"
        assert "is_valid" in data
        assert "results" in data
        
    def test_builder_workflow(self, api_client):
        """Test workflow using client builders."""
        # Create request using builder
        request = create_arganno_request("Test argument content", "Builder test").build()
        
        # Send to API
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        # Verify response structure
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "verification_data" in data
        assert "executed_handlers" in data
        
    def test_error_handling_workflow(self, api_client):
        """Test error handling in end-to-end workflow."""
        # Invalid request (missing required fields)
        invalid_request = {}  # Missing inputs field entirely
        
        response = api_client.post("/api/v1/verify/arganno", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
        assert data["detail"][0]["type"] == "missing"


class TestMultiVerifierWorkflows:
    """Test workflows across multiple verifiers."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_sequential_verifier_calls(self, mock_registry, api_client, sample_request_data, sample_infreco_request_data):
        """Test calling multiple verifiers sequentially."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = ["TestHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        # Test arganno first
        response1 = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response1.status_code == 200
        assert response1.json()["verifier"] == "arganno"
        
        # Test infreco second
        response2 = api_client.post("/api/v1/verify/infreco", json=sample_infreco_request_data)
        assert response2.status_code == 200
        assert response2.json()["verifier"] == "infreco"
        
    def test_different_verifier_configs(self, api_client, sample_request_data, sample_infreco_request_data, sample_logreco_request_data):
        """Test that different verifiers handle their specific configs correctly."""
        # Test arganno (no config options)
        response1 = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response1.status_code == 200
        
        # Test infreco (from_key config)
        response2 = api_client.post("/api/v1/verify/infreco", json=sample_infreco_request_data)
        assert response2.status_code == 200
        
        # Test logreco (from_key and formalization_key configs)
        response3 = api_client.post("/api/v1/verify/logreco", json=sample_logreco_request_data)
        assert response3.status_code == 200