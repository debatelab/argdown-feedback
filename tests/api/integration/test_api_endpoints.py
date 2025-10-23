"""Integration tests for FastAPI endpoints."""

from unittest.mock import patch, MagicMock


class TestRootEndpoint:
    """Test root endpoint functionality."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns correct response."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
        
    def test_root_endpoint_headers(self, api_client):
        """Test root endpoint returns correct headers."""
        response = api_client.get("/")
        assert response.headers["content-type"] == "application/json"


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint(self, api_client):
        """Test health endpoint returns healthy status."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    def test_health_endpoint_details(self, api_client):
        """Test health endpoint includes system details."""
        response = api_client.get("/health")
        data = response.json()
        assert "version" in data
        assert "service" in data


class TestVerifiersEndpoint:
    """Test verifiers listing endpoint."""
    
    def test_list_verifiers(self, api_client):
        """Test listing available verifiers."""
        response = api_client.get("/api/v1/verifiers")
        assert response.status_code == 200
        data = response.json()
        
        # Should have the VerifiersList structure
        assert "core" in data
        assert "coherence" in data
        assert "content_check" in data
        assert isinstance(data["core"], list)
        assert isinstance(data["coherence"], list)
        assert isinstance(data["content_check"], list)
        
    def test_verifiers_include_all_types(self, api_client):
        """Test that all expected verifier types are listed."""
        response = api_client.get("/api/v1/verifiers")
        data = response.json()
        
        # Collect all verifier names from all categories
        all_verifiers = []
        for category in ["core", "coherence", "content_check"]:
            if category in data:
                all_verifiers.extend([v["name"] for v in data[category]])
        
        # Should include core verifier types (processing is not available)
        expected_types = ["arganno", "argmap", "infreco", "logreco"]
        for expected in expected_types:
            assert expected in all_verifiers
        
        # Should also have some coherence and content check verifiers
        assert len(all_verifiers) >= 10  # At least the ones we saw in the output


class TestVerifyEndpoint:
    """Test verification endpoint functionality."""
    
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_verify_with_mock_handler(self, mock_registry, api_client, sample_request_data):
        """Test verification with mocked handler."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.request_id = "test-123"
        mock_result.results = []
        mock_result.executed_handlers = ["TestHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler

        # Make request
        response = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response.status_code == 200

        # Verify handler was called
        mock_registry.create_handler.assert_called_once_with("arganno", filters=None)
        mock_handler.process.assert_called_once()
        
    def test_verify_invalid_verifier(self, api_client, sample_request_data):
        """Test verification with invalid verifier name."""
        response = api_client.post("/api/v1/verify/invalid_verifier", json=sample_request_data)
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "verifier_not_found"
        assert "invalid_verifier" in data["detail"]["message"]
        
    def test_verify_invalid_request_data(self, api_client):
        """Test verification with invalid request data."""
        invalid_data = {"invalid": "data"}
        response = api_client.post("/api/v1/verify/arganno", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
    def test_verify_missing_inputs(self, api_client):
        """Test verification with missing required inputs."""
        incomplete_data = {
            "source": "test source",
            "config": {}
        }
        response = api_client.post("/api/v1/verify/arganno", json=incomplete_data)
        assert response.status_code == 422
        
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_verify_response_format(self, mock_registry, api_client, sample_request_data):
        """Test verification response has correct format."""
        # Setup mock handler
        mock_handler = MagicMock()
        mock_result = MagicMock()
        mock_result.results = []
        mock_result.executed_handlers = ["TestHandler"]
        mock_handler.process.return_value = mock_result
        mock_registry.create_handler.return_value = mock_handler
        
        response = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response.status_code == 200
        
        # Validate response structure
        data = response.json()
        assert "verifier" in data
        assert "is_valid" in data
        assert "results" in data
        assert "executed_handlers" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["executed_handlers"], list)


class TestErrorHandling:
    """Test API error handling."""
    
    def test_method_not_allowed(self, api_client):
        """Test method not allowed error."""
        response = api_client.put("/")  # PUT not allowed on root
        assert response.status_code == 405
        
    def test_not_found_endpoint(self, api_client):
        """Test 404 for non-existent endpoint."""
        response = api_client.get("/nonexistent")
        assert response.status_code == 404
        
    @patch('argdown_feedback.api.server.services.verification_service.verifier_registry')
    def test_internal_server_error_handling(self, mock_registry, api_client, sample_request_data):
        """Test handling of internal server errors."""
        # Setup mock to raise exception
        mock_registry.create_handler.side_effect = Exception("Test error")
        
        response = api_client.post("/api/v1/verify/arganno", json=sample_request_data)
        assert response.status_code == 400  # API converts internal errors to 400
        data = response.json()
        assert "detail" in data


class TestRequestValidation:
    """Test request validation and serialization."""
    
    def test_valid_request_serialization(self, api_client, sample_verification_request):
        """Test that valid requests are properly serialized."""
        request_data = sample_verification_request.model_dump()
        
        with patch('argdown_feedback.api.server.services.verifier_registry') as mock_registry:
            mock_handler = MagicMock()
            mock_result = MagicMock()
            mock_result.request_id = "test-123"
            mock_result.results = []
            mock_result.executed_handlers = []
            mock_handler.process.return_value = mock_result
            mock_registry.create_handler.return_value = mock_handler
            
            response = api_client.post("/api/v1/verify/arganno", json=request_data)
            assert response.status_code == 200
            
    def test_request_with_config(self, api_client, sample_infreco_request_data):
        """Test request with configuration parameters."""
        # Use infreco which accepts config options
        request_data = sample_infreco_request_data.copy()
        
        with patch('argdown_feedback.api.server.services.verification_service.verifier_registry') as mock_registry:
            mock_handler = MagicMock()
            mock_result = MagicMock()
            mock_result.results = []
            mock_result.executed_handlers = []
            mock_handler.process.return_value = mock_result
            mock_registry.create_handler.return_value = mock_handler
            
            response = api_client.post("/api/v1/verify/infreco", json=request_data)
            assert response.status_code == 200