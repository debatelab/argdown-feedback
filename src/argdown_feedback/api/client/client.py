"""
Verification client with pluggable backend support.

Provides a unified interface for verification operations with support for multiple
backends (HTTP, in-process, etc.) while maintaining backwards compatibility.
"""

import asyncio
import warnings
from typing import Union, Optional

from .backends.base import VerificationBackend
from ..shared.models import VerificationRequest, VerificationResponse, VerifierInfo, VerifiersList


class VerifiersClient:
    """
    Generic verification client supporting multiple backends.
    
    This client provides a unified interface for verification operations while
    supporting different backend implementations (HTTP, in-process, etc.).
    
    Supports both synchronous and asynchronous operation modes.
    
    Examples:
        # Recommended: Explicit HTTP backend
        >>> from argdown_feedback.api.client.backends import HTTPBackend
        >>> client = VerifiersClient(backend=HTTPBackend("http://localhost:8000"))
        
        # Recommended: In-process backend (no server needed)
        >>> from argdown_feedback.api.client.backends import InProcessBackend
        >>> client = VerifiersClient(backend=InProcessBackend())
        
        # Backwards compatible (deprecated)
        >>> client = VerifiersClient(base_url="http://localhost:8000")
    """
    
    def __init__(
        self, 
        backend_or_url: Optional[Union[VerificationBackend, str]] = None,
        base_url: Optional[str] = None,
        *,  # Force keyword-only arguments after base_url for other params
        async_client: bool = True,
        timeout: float = 30.0,
        backend: Optional[VerificationBackend] = None  # New param for explicit backend
    ):
        """
        Initialize the verifiers client.
        
        Args:
            backend_or_url: Either a VerificationBackend instance or a base_url string.
                           Recommended: Pass VerificationBackend instance.
                           Deprecated: Pass base_url string (for backwards compatibility).
            base_url: [DEPRECATED] Explicit base_url (alternative to first positional arg).
                     Use backend=HTTPBackend(base_url) instead.
            async_client: Whether to prefer async operations (default: True)
            timeout: [DEPRECATED] Request timeout - only used with base_url.
                    Pass to HTTPBackend constructor instead.
            backend: [DEPRECATED] Explicit backend keyword argument.
                    Use first positional arg instead.
        
        Raises:
            ValueError: If neither backend nor base_url is provided
        
        Examples:
            # Explicit backend (recommended)
            >>> from argdown_feedback.api.client.backends import HTTPBackend
            >>> client = VerifiersClient(HTTPBackend("http://localhost:8000", timeout=60.0))
            
            # In-process backend
            >>> from argdown_feedback.api.client.backends import InProcessBackend
            >>> client = VerifiersClient(InProcessBackend(max_workers=8))
            
            # Backwards compatible
            >>> client = VerifiersClient("http://localhost:8000")  # Shows deprecation warning
        """
        # Handle backwards compatibility for multiple ways of specifying backend/URL
        actual_backend: Optional[VerificationBackend] = None
        
        # Priority 1: Explicit backend keyword argument (deprecated but supported)
        if backend is not None:
            warnings.warn(
                "Passing 'backend' as keyword argument is deprecated. "
                "Pass backend as first positional argument instead: VerifiersClient(backend)",
                DeprecationWarning,
                stacklevel=2
            )
            actual_backend = backend
        
        # Priority 2: First positional argument
        if backend_or_url is not None:
            if isinstance(backend_or_url, str):
                # String passed - treat as base_url for backwards compatibility
                if base_url is not None:
                    raise ValueError("Cannot specify base_url in both positional and keyword arguments")
                
                warnings.warn(
                    "Passing 'base_url' as positional argument is deprecated. "
                    "Use VerifiersClient(backend=HTTPBackend(base_url)) instead. "
                    "This compatibility mode will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2
                )
                from .backends.http import HTTPBackend
                actual_backend = HTTPBackend(backend_or_url, timeout)
            else:
                # VerificationBackend instance passed
                if actual_backend is not None:
                    raise ValueError("Cannot specify backend in both positional and keyword arguments")
                actual_backend = backend_or_url
        
        # Priority 3: base_url keyword argument
        if base_url is not None:
            if actual_backend is not None:
                raise ValueError(
                    "Cannot specify both 'backend' and 'base_url'. "
                    "Use VerifiersClient(HTTPBackend(base_url)) instead."
                )
            
            warnings.warn(
                "Passing 'base_url' as keyword argument is deprecated. "
                "Use VerifiersClient(HTTPBackend(base_url)) instead. "
                "This compatibility mode will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            from .backends.http import HTTPBackend
            actual_backend = HTTPBackend(base_url, timeout)
        
        # Final check
        if actual_backend is None:
            raise ValueError(
                "Must provide either a backend or base_url. "
                "Recommended: VerifiersClient(HTTPBackend(base_url)) or VerifiersClient(InProcessBackend())"
            )
        
        self.backend = actual_backend
        self.is_async = async_client
    
    async def verify_async(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        """
        Verify code asynchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response
            
        Raises:
            RuntimeError: If client was initialized in sync mode
            Backend-specific exceptions: Depends on backend implementation
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        return await self.backend.verify_async(verifier_name, request)
    
    def verify_sync(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        """
        Verify code synchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response
            
        Raises:
            RuntimeError: If client was initialized in async mode
            Backend-specific exceptions: Depends on backend implementation
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        return self.backend.verify_sync(verifier_name, request)
    
    def verify(self, verifier_name: str, request: VerificationRequest) -> Union[VerificationResponse, asyncio.Task]:
        """
        Verify code using the appropriate mode (sync/async).
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response (sync) or Task (async)
        """
        if self.is_async:
            return asyncio.create_task(self.verify_async(verifier_name, request))
        else:
            return self.verify_sync(verifier_name, request)
    
    async def list_verifiers_async(self) -> VerifiersList:
        """
        List all available verifiers asynchronously.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            RuntimeError: If client was initialized in sync mode
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        return await self.backend.list_verifiers_async()
    
    def list_verifiers_sync(self) -> VerifiersList:
        """
        List all available verifiers synchronously.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            RuntimeError: If client was initialized in async mode
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        return self.backend.list_verifiers_sync()
    
    async def get_verifier_info_async(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier asynchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information
            
        Raises:
            RuntimeError: If client was initialized in sync mode
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        return await self.backend.get_verifier_info_async(verifier_name)
    
    def get_verifier_info_sync(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier synchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information
            
        Raises:
            RuntimeError: If client was initialized in async mode
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        return self.backend.get_verifier_info_sync(verifier_name)
    
    def close(self):
        """Close the backend and release resources."""
        self.backend.close()
    
    async def aclose(self):
        """Asynchronously close the backend and release resources."""
        await self.backend.aclose()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()