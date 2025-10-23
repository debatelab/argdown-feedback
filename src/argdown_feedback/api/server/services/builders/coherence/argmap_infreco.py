from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)

from ..base import VerifierBuilder
from .....shared.models import VerifierConfigOption


class ArgmapInfrecoBuilder(VerifierBuilder):
    """Builder for argument map and informal reconstruction coherence verifier."""
    
    name = "argmap_infreco"
    description = "Checks coherence between argument maps and informal argument reconstructions"
    input_types = ["argdown"]
    allowed_filter_roles = ["argmap", "infreco"]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False
        )
    ]
    is_coherence_verifier = True
    
    def build_handlers_pipeline(self, filters_spec: dict, **kwargs) -> List[BaseHandler]:
        """Build argmap_infreco coherence verification pipeline."""

        # default filters
        if "argmap" not in filters_spec:
            filters_spec["argmap"] = [{"key": "filename", "value": "map.*", "regex": True}]
        if "infreco" not in filters_spec:
            filters_spec["infreco"] = [{"key": "filename", "value": "reconstruction.*", "regex": True}]

        # Filters for the verification data. 
        vd_filters = self._create_vd_filters(filters_spec)
        argmap_filter = vd_filters["argmap"]
        infreco_filter = vd_filters["infreco"]
        # The first filter is applied to extract argdown map,
        # and the second to extract the argdown reco.
        filters = (argmap_filter, infreco_filter)

        # Create custom InfRecoCompositeHandler 
        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler", filter=infreco_filter),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=infreco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=infreco_filter),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=infreco_filter),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=infreco_filter),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=infreco_filter),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", filter=infreco_filter),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", filter=infreco_filter),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", filter=infreco_filter),
            ],
            filter=infreco_filter,
            **kwargs
        )
        # Keep UsesAllPropsHandler for coherence verification (unlike standalone infreco)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(name="HasArgdownHandler.map", filter=argmap_filter),
            HasArgdownHandler(name="HasArgdownHandler.reco", filter=infreco_filter),
            ArgMapCompositeHandler(filter=argmap_filter),
            infreco_handler,
            ArgmapInfrecoCoherenceHandler(filters=filters, **kwargs)
        ]

