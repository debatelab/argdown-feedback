"""
Tests for InProcessBackend implementation.

TODO: Implement comprehensive test suite covering:
- Successful verification requests (sync and async)
- Listing verifiers (sync and async)
- Getting verifier info (sync and async)
- Error handling (verifier not found, verification errors)
- Thread pool behavior
- Reuse of VerificationService
- Comparison with HTTP backend for identical behavior
"""

from argdown_feedback.api.client.backends.inprocess import InProcessBackend


class TestInProcessBackend:
    """Test suite for InProcessBackend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = InProcessBackend(max_workers=8)
        assert backend.verification_service is not None
        assert backend._executor is not None
        backend.close()
    
    def test_default_max_workers(self):
        """Test default max_workers value."""
        backend = InProcessBackend()
        # Should use default of 4
        backend.close()
    
    # TODO: Add more tests:
    # - test_verify_sync_success
    # - test_verify_async_success
    # - test_verify_verifier_not_found
    # - test_list_verifiers_sync
    # - test_list_verifiers_async
    # - test_get_verifier_info_sync
    # - test_get_verifier_info_async
    # - test_verify_error_handling
    # - test_thread_pool_cleanup
    # - test_identical_behavior_to_http (integration test)
