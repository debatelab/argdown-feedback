"""
Tests for HTTPBackend implementation.

TODO: Implement comprehensive test suite covering:
- Successful verification requests (sync and async)
- Listing verifiers (sync and async)  
- Getting verifier info (sync and async)
- Error handling (404, 500, network errors)
- Timeout handling
- Context manager support
- Connection pooling behavior
"""

from argdown_feedback.api.client.backends.http import HTTPBackend


class TestHTTPBackend:
    """Test suite for HTTPBackend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = HTTPBackend(base_url="http://localhost:8000", timeout=60.0)
        assert backend.base_url == "http://localhost:8000"
        assert backend.timeout == 60.0
        backend.close()
    
    def test_base_url_normalization(self):
        """Test that trailing slashes are removed from base_url."""
        backend = HTTPBackend(base_url="http://localhost:8000/", timeout=30.0)
        assert backend.base_url == "http://localhost:8000"
        backend.close()
    
    # TODO: Add more tests:
    # - test_verify_sync_success
    # - test_verify_async_success
    # - test_verify_http_error
    # - test_list_verifiers_sync
    # - test_list_verifiers_async
    # - test_get_verifier_info_sync
    # - test_get_verifier_info_async
    # - test_context_manager_sync
    # - test_context_manager_async
    # - test_timeout_handling
    # - test_network_error_handling
