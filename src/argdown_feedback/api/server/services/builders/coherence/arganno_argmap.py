from typing import List

from bs4 import BeautifulSoup
from pyargdown import ArgdownMultiDiGraph
import textdistance

from argdown_feedback.api.server.services.verifier_registry import BaseScorer
from argdown_feedback.api.shared.models import ScoringResult
from argdown_feedback.tasks.base import Evaluation
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
from argdown_feedback.verifiers.verification_request import VerificationRequest

from ..base import VerifierBuilder
from ..core.arganno import (
    AnnotationCoverageScorer,
    AnnotationDensityScorer,
    AnnotationScopeScorer,
)


### Scorers ###

# NOTE
# Scorers subclassed from core.arganno module
# works out of the box with verifier `arganno_infreco`,
# because there is just one `soup` artifact in the
# verification request produced by the handler pipeline.


class AnnotationArgmapScopeScorer(AnnotationScopeScorer):
    """Scorer that evaluates the number of text elements annotated."""

    scorer_id = "annotation_infreco_scope_scorer"


class AnnotationArgmapDensityScorer(AnnotationDensityScorer):
    """Scorer that evaluates the density of dialectical relations identified in the annotation."""

    scorer_id = "annotation_infreco_density_scorer"


class AnnotationArgmapCoverageScorer(AnnotationCoverageScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_infreco_coverage_scorer"


class AnnotationArgmapSemanticCoherenceScorer(BaseScorer):
    """Scorer that evaluates the semantic coherence between argumentative annotations and argument maps."""

    scorer_id = "annotation_argmap_semantic_coherence_scorer"
    scorer_description = "Scores the semantic coherence between argumentative annotations and argument maps."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        soup = evaluation.artifacts.get("soup")
        argdown = evaluation.artifacts.get("argdown_map")
        if not isinstance(soup, BeautifulSoup) or not isinstance(argdown, ArgdownMultiDiGraph):
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No XML content or argument map found for semantic coherence scoring.",
                score=0.0,
                details={},
            )

        dlss: list[float] = []
        for anno_prop in soup.find_all("proposition"):
            anno_label = anno_prop.get("argument_label")  # type: ignore
            anno_text = anno_prop.get_text()  # type: ignore
            ad_prop = next(
                (p for p in argdown.propositions if p.label == anno_label), None
            )
            if ad_prop and anno_text:
                for text in ad_prop.texts:
                    dlss.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text, anno_text
                        )
                    )
            ad_arg = next((a for a in argdown.arguments if a.label == anno_label), None)
            if ad_arg and anno_text:
                for text in ad_arg.gists:
                    dlss.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text, anno_text
                        )
                    )

        score = round(sum(dlss) / len(dlss), 1) if dlss else 0.0

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Semantic coherence score between annotations and argument map: {score:.2f}.",
            score=score,
            details={"pairwise_similarities": dlss},
        )
        return scoring


### Verifier Builder ###

class ArgannoArgmapBuilder(VerifierBuilder):
    """Builder for argumentative annotation and argument map coherence verifier."""

    name = "arganno_argmap"
    description = "Checks coherence between argumentative annotations and argument maps"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "argmap"]
    scorer_classes = [
        AnnotationArgmapSemanticCoherenceScorer,
        AnnotationArgmapCoverageScorer,
        AnnotationArgmapDensityScorer,
        AnnotationArgmapScopeScorer,
    ]
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
