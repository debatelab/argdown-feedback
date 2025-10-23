from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    InfRecoCompositeHandler,
    NoPropInlineDataHandler,
)
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)

from ..base import VerifierBuilder
from .....shared.models import VerifierConfigOption


class LogrecoBuilder(VerifierBuilder):
    """Builder for logical argument reconstruction verifier."""
    
    name = "logreco"
    description = "Validates logical argument reconstruction in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["logreco"]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False
        ),
        VerifierConfigOption(
            name="formalization_key",
            type="string",
            default="formalization",
            description="Key used for formalization information",
            required=False
        ),
        VerifierConfigOption(
            name="declarations_key",
            type="string",
            default="declarations",
            description="Key used for declarations information",
            required=False
        )
    ]
    
    def build_handlers_pipeline(self, filters_spec: dict, **kwargs) -> List[BaseHandler]:
        """Build logreco verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)
        
        # Create InfRecoCompositeHandler and remove NoPropInlineDataHandler
        infreco_handler = InfRecoCompositeHandler(filter=vd_filters.get("logreco"), **{k:v for k,v in kwargs.items() if k == "from_key"})
        infreco_handler.handlers = [
            h for h in infreco_handler.handlers
            if not isinstance(h, NoPropInlineDataHandler)
        ]
        
        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(filter=vd_filters.get("logreco")),
            infreco_handler,
            LogRecoCompositeHandler(filter=vd_filters.get("logreco"), **kwargs)
        ]
