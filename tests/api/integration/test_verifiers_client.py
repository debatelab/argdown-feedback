"""Tests for VerifiersClient high-level API client."""

import pytest
from unittest.mock import patch, MagicMock

from argdown_feedback.api.client.client import VerifiersClient
from argdown_feedback.api.shared.models import VerificationRequest, VerificationResponse


@pytest.fixture
def mock_server_url():
    """Mock server URL for testing."""
    return "http://testserver"


@pytest.fixture
def sync_client(mock_server_url):
    """Create a synchronous VerifiersClient for testing."""
    return VerifiersClient(mock_server_url, async_client=False)


@pytest.fixture
def async_client(mock_server_url):
    """Create an asynchronous VerifiersClient for testing."""
    return VerifiersClient(mock_server_url, async_client=True)


class TestVerifiersClientSync:
    """Test synchronous VerifiersClient.verify() methods."""
    
    def test_verify_sync_basic(self, sync_client):
        """Test basic synchronous verification request."""
        # Create a simple request
        request = VerificationRequest(
            inputs="""```argdown
<Arg>: Test argument.
(1) First premise
-- {from: ["1"]} --
(2) Conclusion
```""",
            source=None,
            config={"from_key": "from"}
        )
        
        # Mock the HTTP response
        with patch.object(sync_client.backend._sync_client, 'post') as mock_post:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "verifier": "infreco",
                "is_valid": True,
                "verification_data": [],
                "results": [],
                "scores": [],
                "executed_handlers": ["InfRecoHandler"],
                "processing_time_ms": 42.5
            }
            mock_post.return_value = mock_response
            
            # Call verify_sync
            response = sync_client.verify_sync("infreco", request)
            
            # Verify the call was made correctly
            mock_post.assert_called_once_with(
                "http://testserver/api/v1/verify/infreco",
                json=request.dict()
            )
            
            # Verify response
            assert isinstance(response, VerificationResponse)
            assert response.verifier == "infreco"
            assert response.is_valid
            assert response.processing_time_ms == 42.5
    
    def test_verify_delegates_to_verify_sync(self, sync_client):
        """Test that verify() delegates to verify_sync() for sync client."""
        request = VerificationRequest(
            inputs="Test input",
            source=None,
            config={}
        )
        
        with patch.object(sync_client, 'verify_sync') as mock_verify_sync:
            mock_verify_sync.return_value = MagicMock(spec=VerificationResponse)
            
            result = sync_client.verify("arganno", request)
            
            mock_verify_sync.assert_called_once_with("arganno", request)
            assert result is mock_verify_sync.return_value
    
    def test_verify_sync_multiple_verifiers(self, sync_client):
        """Test calling multiple verifiers sequentially."""
        request = VerificationRequest(
            inputs="```xml\n<proposition id='1'>Test</proposition>\n```",
            source=None,
            config={}
        )
        
        with patch.object(sync_client.backend._sync_client, 'post') as mock_post:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "verifier": "test",
                "is_valid": True,
                "verification_data": [],
                "results": [],
                "scores": [],
                "executed_handlers": [],
                "processing_time_ms": 10.0
            }
            mock_post.return_value = mock_response
            
            # Call multiple verifiers
            verifiers = ["arganno", "argmap", "infreco"]
            responses = []
            
            for verifier in verifiers:
                response = sync_client.verify_sync(verifier, request)
                responses.append(response)
            
            # Verify all calls were made
            assert mock_post.call_count == 3
            assert len(responses) == 3
            assert all(isinstance(r, VerificationResponse) for r in responses)
    
    def test_verify_sync_with_filters(self, sync_client):
        """Test verification with custom filters."""
        request = VerificationRequest(
            inputs="Multiple code blocks",
            source=None,
            config={
                "from_key": "premises",
                "filters": {
                    "infreco": {
                        "filename": "reconstruction.ad"
                    }
                }
            }
        )
        
        with patch.object(sync_client.backend._sync_client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "verifier": "infreco",
                "is_valid": True,
                "verification_data": [],
                "results": [],
                "scores": [],
                "executed_handlers": [],
                "processing_time_ms": 15.0
            }
            mock_post.return_value = mock_response
            
            response = sync_client.verify_sync("infreco", request)
            
            # Verify the config was passed correctly
            call_args = mock_post.call_args
            assert call_args[1]['json']['config']['filters']['infreco']['filename'] == "reconstruction.ad"
            assert response.is_valid


class TestVerifiersClientAsync:
    """Test asynchronous VerifiersClient.verify() methods."""
    
    @pytest.mark.asyncio
    async def test_verify_async_basic(self, async_client):
        """Test basic asynchronous verification request."""
        request = VerificationRequest(
            inputs="""```argdown
<Arg>: Async test.
(1) Premise
```""",
            source=None,
            config={}
        )
        
        with patch.object(async_client.backend._async_client, 'post') as mock_post:
            # Setup async mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "verifier": "argmap",
                "is_valid": True,
                "verification_data": [],
                "results": [],
                "scores": [],
                "executed_handlers": ["ArgmapHandler"],
                "processing_time_ms": 25.0
            }
            # Make it awaitable
            async def async_post(*args, **kwargs):
                return mock_response
            mock_post.side_effect = async_post
            
            # Call verify_async
            response = await async_client.verify_async("argmap", request)
            
            # Verify response
            assert isinstance(response, VerificationResponse)
            assert response.verifier == "argmap"
            assert response.is_valid
            assert response.processing_time_ms == 25.0
    
    @pytest.mark.asyncio
    async def test_verify_async_error_handling(self, async_client):
        """Test async client handles HTTP errors correctly."""
        request = VerificationRequest(
            inputs="Test",
            source=None,
            config={}
        )
        
        with patch.object(async_client.backend._async_client, 'post') as mock_post:
            # Setup error response
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("Not found")
            
            async def async_post(*args, **kwargs):
                return mock_response
            mock_post.side_effect = async_post
            
            # Verify exception is raised
            with pytest.raises(Exception, match="Not found"):
                await async_client.verify_async("invalid_verifier", request)


class TestVerifiersClientDiscovery:
    """Test VerifiersClient discovery methods."""
    
    def test_list_verifiers_sync(self, sync_client):
        """Test synchronous verifier listing."""
        with patch.object(sync_client.backend._sync_client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "core": [
                    {"name": "arganno", "description": "Test", "input_types": ["xml"], 
                     "allowed_filter_roles": ["arganno"], "config_options": [], "is_coherence_verifier": False}
                ],
                "coherence": [],
                "content_check": []
            }
            mock_get.return_value = mock_response
            
            result = sync_client.list_verifiers_sync()
            
            mock_get.assert_called_once_with("http://testserver/api/v1/verifiers")
            assert len(result.core) == 1
            assert result.core[0].name == "arganno"
    
    def test_get_verifier_info_sync(self, sync_client):
        """Test getting info about a specific verifier."""
        with patch.object(sync_client.backend._sync_client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "infreco",
                "description": "Validates informal reconstructions",
                "input_types": ["argdown"],
                "allowed_filter_roles": ["infreco"],
                "config_options": [
                    {
                        "name": "from_key",
                        "type": "string",
                        "default": "from",
                        "description": "Key for inference info",
                        "required": False
                    }
                ],
                "is_coherence_verifier": False
            }
            mock_get.return_value = mock_response
            
            result = sync_client.get_verifier_info_sync("infreco")
            
            mock_get.assert_called_once_with("http://testserver/api/v1/verifiers/infreco")
            assert result.name == "infreco"
            assert len(result.config_options) == 1
            assert result.config_options[0].name == "from_key"


class TestVerifiersClientContextManager:
    """Test VerifiersClient context manager support."""
    
    def test_sync_context_manager(self, mock_server_url):
        """Test sync client works as context manager."""
        with VerifiersClient(mock_server_url, async_client=False) as client:
            assert client is not None
            assert isinstance(client, VerifiersClient)
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_server_url):
        """Test async client works as async context manager."""
        async with VerifiersClient(mock_server_url, async_client=True) as client:
            assert client is not None
            assert isinstance(client, VerifiersClient)
