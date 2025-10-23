from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    InfRecoCompositeHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)

from ..base import VerifierBuilder
from .....shared.models import VerifierConfigOption


class InfrecoBuilder(VerifierBuilder):
    """Builder for informal argument reconstruction verifier."""
    
    name = "infreco"
    description = "Validates informal argument reconstruction in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["infreco"]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False
        )
    ]
    
    def build_handlers_pipeline(self, filters_spec: dict, **kwargs) -> List[BaseHandler]:
        """Build infreco verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)
        
        # Create InfRecoCompositeHandler and remove UsesAllPropsHandler
        infreco_handler = InfRecoCompositeHandler(filter=vd_filters.get("infreco"), **kwargs)
        infreco_handler.handlers = [
            h for h in infreco_handler.handlers
            if not isinstance(h, UsesAllPropsHandler)
        ]
        
        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(filter=vd_filters.get("infreco")),
            infreco_handler
        ]

