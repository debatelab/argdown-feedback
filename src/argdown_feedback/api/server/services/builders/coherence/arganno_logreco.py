from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
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
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.arganno_logreco_handler import (
    ArgannoLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)

from ..base import VerifierBuilder
from .....shared.models import VerifierConfigOption

from ..core.arganno import (
    AnnotationCoverageScorer,
    AnnotationDensityScorer,
    AnnotationScopeScorer,
)
from ..core.logreco import (
    LogrecoFormalizationsFaithfulnessScorer,
    LogrecoPredicateLogicScorer,
    LogrecoTrivialityScorer,
)
from .arganno_infreco import (
    AnnotationInfrecoSemanticCoherenceScorer,
)


### Scorers ###

# NOTE
# Scorers subclassed from other modules
# work out of the box with verifier `arganno_infreco`,
# because there is just one `soup` artifact in the
# verification request produced by the handler pipeline.


class AnnotationLogrecoScopeScorer(AnnotationScopeScorer):
    """Scorer that evaluates the number of text elements annotated."""

    scorer_id = "annotation_logreco_scope_scorer"


class AnnotationLogrecoDensityScorer(AnnotationDensityScorer):
    """Scorer that evaluates the density of dialectical relations identified in the annotation."""

    scorer_id = "annotation_logreco_density_scorer"


class AnnotationLogrecoCoverageScorer(AnnotationCoverageScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_logreco_coverage_scorer"


class AnnotationLogrecoFormalizationsFaithfulnessScorer(
    LogrecoFormalizationsFaithfulnessScorer
):
    """Scorer that evaluates the faithfulness of logical formalizations to the foamlized propositions."""

    scorer_id = "annotation_logreco_formalizations_faithfulness_scorer"


class AnnotationLogrecoPredicateLogicScorer(LogrecoPredicateLogicScorer):
    """Scorer that evaluates the use of predicate logic formalizations."""

    scorer_id = "annotation_logreco_predicate_logic_scorer"


class AnnotationLogrecoTrivialityScorer(LogrecoTrivialityScorer):
    """Scorer that evaluates the triviality of logical formalizations."""

    scorer_id = "annotation_logreco_triviality_scorer"


class AnnotationLogrecoSemanticCoherenceScorer(
    AnnotationInfrecoSemanticCoherenceScorer
):
    """Scorer that evaluates the semantic coherence between argumentative annotations and informal reconstructions."""

    scorer_id = "annotation_logreco_semantic_coherence_scorer"



### Verifier Builder  ###

class ArgannoLogrecoBuilder(VerifierBuilder):
    """Builder for argumentative annotation and logical reconstruction coherence verifier."""

    name = "arganno_logreco"
    description = "Checks coherence between argumentative annotations and logical argument reconstructions"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "logreco"]
    scorer_classes = [
        AnnotationLogrecoScopeScorer,
        AnnotationLogrecoDensityScorer,
        AnnotationLogrecoCoverageScorer,
        AnnotationLogrecoFormalizationsFaithfulnessScorer,
        AnnotationLogrecoPredicateLogicScorer,
        AnnotationLogrecoTrivialityScorer,
        AnnotationLogrecoSemanticCoherenceScorer,
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
    ]
    is_coherence_verifier = True

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build arganno_logreco coherence verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        if "arganno" in vd_filters and "logreco" in vd_filters:
            arganno_filter = vd_filters["arganno"]
            logreco_filter = vd_filters["logreco"]
            # Filters for the verification data.
            # The first filter is applied to extract argdown reco,
            # and the second to extract the xml annotation.
            # If None, default filters are used.
            filters = (logreco_filter, arganno_filter)
        else:
            arganno_filter = None
            logreco_filter = None
            filters = None

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler"),
                HasPCSHandler(name="InfReco.HasPCSHandler"),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler"),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler"),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler"),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler"),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler"),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler"),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler"),
            ],
            filter=logreco_filter,
            **{k: v for k, v in kwargs.items() if k in ["from_key"]},
        )

        return [
            DefaultProcessingHandler(),
            HasAnnotationsHandler(filter=arganno_filter),
            HasArgdownHandler(filter=logreco_filter),
            ArgannoCompositeHandler(filter=arganno_filter),
            infreco_handler,
            LogRecoCompositeHandler(filter=logreco_filter, **kwargs),
            ArgannoLogrecoCoherenceHandler(
                filters=filters,
                **{k: v for k, v in kwargs.items() if k in ["from_key"]},
            ),
        ]

