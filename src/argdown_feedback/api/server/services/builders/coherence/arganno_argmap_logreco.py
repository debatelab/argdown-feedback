from typing import Any, List

from argdown_feedback.api.shared.filtering import FilterRoleType
from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
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
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    ArgmapLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)

from .....shared.models import VerifierConfigOption

from ..base import VerifierBuilder
from ..core.arganno import (
    AnnotationScopeScorer,
    AnnotationDensityScorer,
    AnnotationCoverageScorer,
)


### Scorers ###


# NOTE
# Scorers subclassed from core.arganno module
# works out of the box with verifier `arganno_infreco`,
# because there is just one `soup` artifact in the
# verification request produced by the handler pipeline.


class AnnotationArgmapLogrecoScopeScorer(AnnotationScopeScorer):
    """Scorer that evaluates the number of text elements annotated."""

    scorer_id = "annotation_argmap_logreco_scope_scorer"


class AnnotationArgmapLogrecoDensityScorer(AnnotationDensityScorer):
    """Scorer that evaluates the density of dialectical relations identified in the annotation."""

    scorer_id = "annotation_argmap_logreco_density_scorer"


class AnnotationArgmapLogrecoCoverageScorer(AnnotationCoverageScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_argmap_logreco_coverage_scorer"




### Verifier Builder ###

class ArgannoArgmapLogrecoBuilder(VerifierBuilder):
    """Builder for argumentative annotation, argument map, and logical reconstruction coherence verifier."""

    name = "arganno_argmap_logreco"
    description = "Checks coherence between argumentative annotations, argument maps, and logical argument reconstructions"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "argmap", "logreco"]
    scorer_classes = [
        AnnotationArgmapLogrecoScopeScorer,
        AnnotationArgmapLogrecoDensityScorer,
        AnnotationArgmapLogrecoCoverageScorer,
    ]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False,
        ),
        VerifierConfigOption(
            name="formalization_key",
            type="string",
            default="formalization",
            description="Key used for formalization information",
            required=False,
        ),
        VerifierConfigOption(
            name="declarations_key",
            type="string",
            default="declarations",
            description="Key used for declarations information",
            required=False,
        ),
        VerifierConfigOption(
            name="N",
            type="integer",
            default=2,
            description="Minimum number of arguments required in the reconstruction",
            required=False,
        ),
    ]
    is_coherence_verifier = True

    def build_handlers_pipeline(
        self, filters_spec: dict[FilterRoleType, Any], **kwargs
    ) -> List[BaseHandler]:
        """Build arganno_argmap_logreco coherence verification pipeline."""

        # default filters
        if "arganno" not in filters_spec:
            filters_spec["arganno"] = [{"key": "filename", "value": "annotation.*", "regex": True}]
        if "argmap" not in filters_spec:
            filters_spec["argmap"] = [{"key": "filename", "value": "map.*", "regex": True}]
        if "logreco" not in filters_spec:
            filters_spec["logreco"] = [{"key": "filename", "value": "reconstruction.*", "regex": True}]
        
        # Filters for the verification data. 
        vd_filters = self._create_vd_filters(filters_spec)
        arganno_filter = vd_filters["arganno"]
        argmap_filter = vd_filters["argmap"]
        logreco_filter = vd_filters["logreco"]


        # Create custom InfRecoCompositeHandler following the provided pattern
        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasAtLeastNArgumentsHandler(
                    name="InfReco.HasAtLeastNArgumentsHandler",
                    filter=logreco_filter,
                    N=kwargs.get("N", 2),
                ),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=logreco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(
                    name="InfReco.StartsWithPremiseHandler", filter=logreco_filter
                ),
                EndsWithConclusionHandler(
                    name="InfReco.EndsWithConclusionHandler", filter=logreco_filter
                ),
                NoDuplicatePCSLabelsHandler(
                    name="InfReco.NoDuplicatePCSLabelsHandler", filter=logreco_filter
                ),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=logreco_filter),
                # Inference data handlers
                HasInferenceDataHandler(
                    name="InfReco.HasInferenceDataHandler", filter=logreco_filter
                ),
                PropRefsExistHandler(
                    name="InfReco.PropRefsExistHandler", filter=logreco_filter
                ),
                UsesAllPropsHandler(
                    name="InfReco.UsesAllPropsHandler", filter=logreco_filter
                ),
                # Extra material handlers
                NoExtraPropositionsHandler(
                    name="InfReco.NoExtraPropositionsHandler", filter=logreco_filter
                ),
            ],
            filter=logreco_filter,
            **{k: v for k,v in kwargs.items() if k == "from_key"},
        )

        return [
            # Processing
            DefaultProcessingHandler(),
            HasAnnotationsHandler(filter=arganno_filter),
            HasArgdownHandler(name="HasArgdownHandler.map", filter=argmap_filter),
            HasArgdownHandler(name="HasArgdownHandler.reco", filter=logreco_filter),
            # Core
            ArgannoCompositeHandler(filter=arganno_filter),
            ArgMapCompositeHandler(filter=argmap_filter),
            infreco_handler,
            LogRecoCompositeHandler(filter=logreco_filter, **kwargs),
            # Coherence
            ArgannoInfrecoCoherenceHandler(
                # Filters for the verification data.
                # The first filter is applied to extract argdown reco,
                # and the second to extract the xml annotation.
                filters=(logreco_filter, arganno_filter),
                **{k: v for k,v in kwargs.items() if k == "from_key"}
            ),
            ArgmapInfrecoCoherenceHandler(
                # Filters for the verification data.
                # The first filter is applied to extract argdown map,
                # and the second to extract the argdown reco.
                filters=(argmap_filter, logreco_filter),
                **{k: v for k,v in kwargs.items() if k == "from_key"},
            ),
            ArgmapLogrecoCoherenceHandler(
                # Filters for the verification data.
                # The first filter is applied to extract argdown map,
                # and the second to extract the argdown reco.
                filters=(argmap_filter, logreco_filter),
                **{k: v for k,v in kwargs.items() if k == "from_key"},
            ),
        ]
