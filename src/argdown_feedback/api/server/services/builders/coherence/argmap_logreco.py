from typing import Any, List

from argdown_feedback.api.shared.filtering import FilterRoleType
from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasAtLeastNArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    NoExtraPropositionsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    ArgmapLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)

from ..base import VerifierBuilder
from ..core.argmap import (
    ArgmapSizeScorer,
    ArgmapDensityScorer,
    ArgmapFaithfulnessScorer,
)

from .....shared.models import VerifierConfigOption


### Scorers ###

# NOTE
# Scorers subclassed from core.argmap module will work,
# assuming that the argument map can be identified
# with our default filters, using filename patterns.

class ArgmapLogrecoSizeScorer(ArgmapSizeScorer):
    """Scorer that evaluates the size of the argument map ."""

    scorer_id = "argmap_logreco_size_scorer"

class ArgmapLogrecoDensityScorer(ArgmapDensityScorer):
    """Scorer that evaluates the density of the argument map ."""

    scorer_id = "argmap_logreco_density_scorer"

class ArgmapLogrecoFaithfulnessScorer(ArgmapFaithfulnessScorer):
    """Scorer that evaluates the faithfulness of the argument map ."""

    scorer_id = "argmap_logreco_faithfulness_scorer"


### Verifier Builder ###

class ArgmapLogrecoBuilder(VerifierBuilder):
    """Builder for argument map and logical reconstruction coherence verifier."""
    
    name = "argmap_logreco"
    description = "Checks coherence between argument maps and logical argument reconstructions"
    input_types = ["argdown"]
    allowed_filter_roles = ["argmap", "logreco"]
    scorer_classes = [
        ArgmapLogrecoSizeScorer,
        ArgmapLogrecoDensityScorer,
        ArgmapLogrecoFaithfulnessScorer
    ]
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
        ),
        VerifierConfigOption(
            name="N",
            type="integer",
            default=2,
            description="Minimum number of arguments required in the reconstruction",
            required=False
        )
    ]
    is_coherence_verifier = True
    
    def build_handlers_pipeline(self, filters_spec: dict[FilterRoleType, Any], **kwargs) -> List[BaseHandler]:
        """Build argmap_logreco coherence verification pipeline."""
        # default filters
        if "argmap" not in filters_spec:
            filters_spec["argmap"] = [{"key": "filename", "value": "map.*", "regex": True}]
        if "logreco" not in filters_spec:
            filters_spec["logreco"] = [{"key": "filename", "value": "reconstruction.*", "regex": True}]

        # Filters for the verification data. 
        vd_filters = self._create_vd_filters(filters_spec)
        argmap_filter = vd_filters["argmap"]
        logreco_filter = vd_filters["logreco"]
        # The first filter is applied to extract argdown map,
        # and the second to extract the argdown reco.
        filters = (argmap_filter, logreco_filter)

        # Create custom InfRecoCompositeHandler following argmap_plus_logreco.py pattern
        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasAtLeastNArgumentsHandler(name="InfReco.HasAtLeastNArgumentsHandler", filter=logreco_filter, N=kwargs.get("N",2)),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=logreco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=logreco_filter),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=logreco_filter),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=logreco_filter),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=logreco_filter),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", filter=logreco_filter),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", filter=logreco_filter),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", filter=logreco_filter),
                # Extra material handlers
                NoExtraPropositionsHandler(name="InfReco.NoExtraPropositionsHandler", filter=logreco_filter),
            ],
            filter=logreco_filter,
            **{k: v for k,v in kwargs.items() if k == "from_key"}
        )

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(name="HasArgdownHandler.map", filter=argmap_filter),
            HasArgdownHandler(name="HasArgdownHandler.reco", filter=logreco_filter),
            ArgMapCompositeHandler(filter=argmap_filter),
            infreco_handler,
            LogRecoCompositeHandler(filter=logreco_filter, **kwargs),
            ArgmapInfrecoCoherenceHandler(filters=filters, **{k: v for k,v in kwargs.items() if k == "from_key"}),
            ArgmapLogrecoCoherenceHandler(filters=filters, **{k: v for k,v in kwargs.items() if k == "from_key"})
        ]
