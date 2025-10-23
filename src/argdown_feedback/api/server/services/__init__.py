"""
Server services for argdown-feedback API.

This package contains service classes for handling verification logic.
"""

from .verifier_registry import VerifierRegistry

from .builders.core.arganno import ArgannoBuilder
from .builders.core.argmap import ArgmapBuilder
from .builders.core.infreco import InfrecoBuilder
from .builders.core.logreco import LogrecoBuilder

from .builders.coherence.arganno_argmap import ArgannoArgmapBuilder
from .builders.coherence.arganno_infreco import ArgannoInfrecoBuilder
from .builders.coherence.arganno_logreco import ArgannoLogrecoBuilder
from .builders.coherence.argmap_infreco import ArgmapInfrecoBuilder
from .builders.coherence.argmap_logreco import ArgmapLogrecoBuilder
from .builders.coherence.arganno_argmap_logreco import ArgannoArgmapLogrecoBuilder

# Global registry instance
verifier_registry = VerifierRegistry()

verifier_registry.register("arganno", ArgannoBuilder())
verifier_registry.register("argmap", ArgmapBuilder())
verifier_registry.register("infreco", InfrecoBuilder())
verifier_registry.register("logreco", LogrecoBuilder())
verifier_registry.register("arganno_argmap", ArgannoArgmapBuilder())
verifier_registry.register("arganno_infreco", ArgannoInfrecoBuilder())
verifier_registry.register("arganno_logreco", ArgannoLogrecoBuilder())
verifier_registry.register("argmap_infreco", ArgmapInfrecoBuilder())
verifier_registry.register("argmap_logreco", ArgmapLogrecoBuilder())
verifier_registry.register("arganno_argmap_logreco", ArgannoArgmapLogrecoBuilder())

__all__ = ["verifier_registry", "VerifierRegistry"]