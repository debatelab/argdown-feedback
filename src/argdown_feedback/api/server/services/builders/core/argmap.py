from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)

from ..base import VerifierBuilder

class ArgmapBuilder(VerifierBuilder):
    """Builder for argument map verifier."""

    name = "argmap"
    description = "Validates argument maps in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["argmap"]
    config_options = []  # ArgMapCompositeHandler doesn't accept any config options

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build argmap verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(filter=vd_filters.get("argmap")),
            ArgMapCompositeHandler(filter=vd_filters.get("argmap")),
        ]
