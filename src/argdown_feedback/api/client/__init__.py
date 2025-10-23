"""
Client library for the argdown-feedback API.
"""

from .client import VerifiersClient
from .builders import *

__all__ = [
    "VerifiersClient",
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