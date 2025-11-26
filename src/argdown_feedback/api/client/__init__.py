"""
Client library for the argdown-feedback API.

The client supports multiple backends for flexible deployment:
- HTTPBackend: Communicate with a remote FastAPI server via HTTP
- InProcessBackend: Execute verification handlers directly without HTTP overhead

Examples:
    # HTTP backend (remote server)
    >>> from argdown_feedback.api.client import VerifiersClient
    >>> from argdown_feedback.api.client.backends import HTTPBackend
    >>> client = VerifiersClient(backend=HTTPBackend("http://localhost:8000"))
    
    # In-process backend (no server needed)
    >>> from argdown_feedback.api.client.backends import InProcessBackend
    >>> client = VerifiersClient(backend=InProcessBackend())
    
    # Backwards compatible (deprecated)
    >>> client = VerifiersClient(base_url="http://localhost:8000")
"""

from .client import VerifiersClient
from .builders import *
from . import backends

__all__ = [
    "VerifiersClient",
    "backends",
    # Core verifier request builders
    "create_arganno_request",
    "create_argmap_request", 
    "create_infreco_request",
    "create_logreco_request",
    "create_has_annotations_request",
    "create_has_argdown_request",
    # Coherence verifier request builders
    "create_arganno_argmap_request",
    "create_arganno_infreco_request",
    "create_arganno_logreco_request",
    "create_argmap_infreco_request",
    "create_argmap_logreco_request",
    "create_arganno_argmap_logreco_request",
]