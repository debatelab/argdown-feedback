from typing import Any, List

import textdistance

from argdown_feedback.api.shared.filtering import FilterRoleType
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
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)
from argdown_feedback.verifiers.verification_request import VerificationRequest

from ..base import VerifierBuilder, BaseScorer
from .....shared.models import ScoringResult, VerifierConfigOption

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


class AnnotationInfrecoScopeScorer(AnnotationScopeScorer):
    """Scorer that evaluates the number of text elements annotated."""

    scorer_id = "annotation_infreco_scope_scorer"


class AnnotationInfrecoDensityScorer(AnnotationDensityScorer):
    """Scorer that evaluates the density of dialectical relations identified in the annotation."""

    scorer_id = "annotation_infreco_density_scorer"


class AnnotationInfrecoCoverageScorer(AnnotationCoverageScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_infreco_coverage_scorer"


class AnnotationInfrecoSemanticCoherenceScorer(BaseScorer):
    """Scorer that evaluates the semantic coherence between argumentative annotations and informal reconstructions."""

    scorer_id = "annotation_infreco_semantic_coherence_scorer"
    scorer_description = "Scores the semantic coherence between argumentative annotations and informal reconstructions."
    _reco_filter_roles: list[FilterRoleType] = ["infreco"]

    def score(self, result: VerificationRequest) -> ScoringResult:
        argdown, _ = self.get_argdown(result, roles=self._reco_filter_roles)
        soup, _ = self.get_xml_soup(result)
        anno_props = soup.find_all("proposition") if soup else None

        if anno_props is None or argdown is None:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="Insufficient data for semantic coherence scoring.",
                score=0.0,
                details={},
            )

        matches: list[tuple[str, str]] = []
        for proposition in argdown.propositions:
            for annotation_id in proposition.data.get("annotation_ids", []):
                anno_prop = next(
                    (ap for ap in anno_props if ap.get("id") == annotation_id), None  # type: ignore
                )
                if anno_prop is None:
                    continue
                for text in proposition.texts:
                    matches.append((anno_prop.get_text(), text))

        dlss = [
            textdistance.damerau_levenshtein.normalized_similarity(s, t)
            for s, t in matches
        ]
        score = round(sum(dlss) / len(dlss), 1) if dlss else 0.0

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Semantic coherence score based on {len(dlss)} matched propositions.",
            score=score,
            details={"matched_propositions": len(dlss)},
        )

        return scoring


#### Verifier Builder ###


class ArgannoInfrecoBuilder(VerifierBuilder):
    """Builder for argumentative annotation and informal reconstruction coherence verifier."""

    name = "arganno_infreco"
    description = "Checks coherence between argumentative annotations and informal argument reconstructions"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "infreco"]
    scorer_classes = [
        AnnotationInfrecoScopeScorer,
        AnnotationInfrecoDensityScorer,
        AnnotationInfrecoCoverageScorer,
        AnnotationInfrecoSemanticCoherenceScorer,
    ]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False,
        ),
    ]
    is_coherence_verifier = True

    def build_handlers_pipeline(
        self, filters_spec: dict[FilterRoleType, Any], **kwargs
    ) -> List[BaseHandler]:
        """Build arganno_infreco coherence verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        if "arganno" in vd_filters and "infreco" in vd_filters:
            arganno_filter = vd_filters["arganno"]
            infreco_filter = vd_filters["infreco"]
            # Filters for the verification data.
            # The first filter is applied to extract argdown reco,
            # and the second to extract the xml annotation.
            # If None, default filters are used.
            filters = (infreco_filter, arganno_filter)
        else:
            arganno_filter = None
            infreco_filter = None
            filters = None

        # Create InfRecoCompositeHandler
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
                HasInferenceDataHandler(
                    name="InfReco.HasInferenceDataHandler", **kwargs
                ),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", **kwargs),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", **kwargs),
            ],
            filter=infreco_filter,
            **kwargs,
        )

        return [
            DefaultProcessingHandler(),
            HasAnnotationsHandler(filter=arganno_filter),
            HasArgdownHandler(filter=infreco_filter),
            ArgannoCompositeHandler(filter=arganno_filter),
            infreco_handler,
            ArgannoInfrecoCoherenceHandler(filters=filters, **kwargs),
        ]
