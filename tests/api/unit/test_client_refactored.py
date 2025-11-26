"""
Tests for refactored VerifiersClient with backend support.

TODO: Implement comprehensive test suite covering:
- Client initialization with explicit backend
- Client initialization with base_url (backwards compatibility)
- Deprecation warnings for base_url usage
- Error on both backend and base_url specified
- Delegation to backend methods
- Sync/async mode switching
- Context manager support
"""

from argdown_feedback.api.client import VerifiersClient
from argdown_feedback.api.client.backends import HTTPBackend, InProcessBackend


class TestVerifiersClient:
    """Test suite for refactored VerifiersClient."""
    
    def test_initialization_with_http_backend(self):
        """Test client initialization with explicit HTTPBackend."""
        backend = HTTPBackend(base_url="http://localhost:8000")
        client = VerifiersClient(backend)  # Recommended: positional argument
        assert client.backend is backend
        assert client.is_async is True
        client.close()
    
    def test_initialization_with_inprocess_backend(self):
        """Test client initialization with explicit InProcessBackend."""
        backend = InProcessBackend()
        client = VerifiersClient(backend)  # Recommended: positional argument
        assert client.backend is backend
        assert client.is_async is True
        client.close()
    
    def test_initialization_backwards_compatible(self):
        """Test backwards compatible initialization with base_url."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client = VerifiersClient(base_url="http://localhost:8000")
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "deprecated" in str(w[-1].message).lower()
        
        assert isinstance(client.backend, HTTPBackend)
        client.close()
    
    def test_initialization_error_both_backend_and_base_url(self):
        """Test that error is raised when both backend and base_url are specified."""
        import pytest
        backend = HTTPBackend(base_url="http://localhost:8000")
        with pytest.raises(ValueError, match="Cannot specify both"):
            VerifiersClient(backend=backend, base_url="http://localhost:8000")
        backend.close()
    
    def test_initialization_error_neither_backend_nor_base_url(self):
        """Test that error is raised when neither backend nor base_url is specified."""
        import pytest
        with pytest.raises(ValueError, match="Must provide either"):
            VerifiersClient()
    
    # TODO: Add more tests:
    # - test_verify_delegates_to_backend
    # - test_list_verifiers_delegates_to_backend
    # - test_get_verifier_info_delegates_to_backend
    # - test_sync_mode_enforcement
    # - test_async_mode_enforcement
    # - test_context_manager_sync
    # - test_context_manager_async
