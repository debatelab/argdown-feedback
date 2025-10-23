from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.arganno_argmap_handler import (
    ArgannoArgmapCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)

from ..base import VerifierBuilder


class ArgannoArgmapBuilder(VerifierBuilder):
    """Builder for argumentative annotation and argument map coherence verifier."""

    name = "arganno_argmap"
    description = "Checks coherence between argumentative annotations and argument maps"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "argmap"]
    config_options = []  # ArgannoArgmapCoherenceHandler doesn't accept any config options
    is_coherence_verifier = True

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build arganno_argmap coherence verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        if "arganno" in vd_filters and "argmap" in vd_filters:
            arganno_filter = vd_filters["arganno"]
            argmap_filter = vd_filters["argmap"]
            # Filters for the verification data.
            # The first filter is applied to extract argdown map,
            # and the second to extract the xml annotation.
            # If None, default filters are used.
            filters = (argmap_filter, arganno_filter)
        else:
            arganno_filter = None
            argmap_filter = None
            filters = None

        return [
            DefaultProcessingHandler(),
            HasAnnotationsHandler(filter=arganno_filter),
            HasArgdownHandler(filter=argmap_filter),
            ArgannoCompositeHandler(filter=arganno_filter, **kwargs),
            ArgMapCompositeHandler(filter=argmap_filter, **kwargs),
            ArgannoArgmapCoherenceHandler(filters=filters, **kwargs),
        ]
