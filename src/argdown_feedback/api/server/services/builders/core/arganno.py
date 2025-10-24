from typing import List, Type

from argdown_feedback.tasks.base import Evaluation

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    FencedCodeBlockExtractor,
    XMLParser,
)
from argdown_feedback.verifiers.verification_request import VerificationRequest

from .....shared.models import ScoringResult, VerifierConfigOption
from ..base import BaseScorer, VerifierBuilder


### Verifier builder ###

class ArgannoBuilder(VerifierBuilder):
    """Builder for argumentative annotation verifier."""

    name = "arganno"
    description = "Validates argumentative annotations in XML format"
    input_types = ["xml"]
    allowed_filter_roles = ["arganno"]
    config_options = [
        VerifierConfigOption(
            name="enable_annotation_coverage_scorer",
            type="bool",
            default=False,
            description="Enable scoring of annotation coverage",
            required=False
        )
    ]  
    
    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build arganno verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            XMLParser(name="XMLAnnotationParser"),
            HasAnnotationsHandler(filter=vd_filters.get("arganno")),
            ArgannoCompositeHandler(filter=vd_filters.get("arganno")),
        ]
    
    def build_scorers(self, filters_spec: dict, **kwargs) -> List[BaseScorer]:
        """Build the list of virtue scorers to be used."""
        scorer_classes: List[Type[BaseScorer]] = [AnnotationCoverageScorer]
        scorers = [
            scorer_class(parent_name=self.name) 
            for scorer_class in scorer_classes
            if kwargs.get("enable_" + scorer_class.scorer_id, True)
        ]
        return scorers


### Scorers ###

class AnnotationCoverageScorer(BaseScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_coverage_scorer"
    scorer_description = "Scores the text coverage of the annotation, i.e. the ratio of annotated text to total text length."

    def score(self, result: VerificationRequest) -> ScoringResult:

        evaluation = Evaluation.from_verification_request(result)
        soup = evaluation.artifacts.get("soup")
        if not soup:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No XML content found for annotation coverage scoring.",
                score=0.0,
                details={},
            )
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        coverage = sum(len(proposition.get_text()) for proposition in propositions)
        text_length = len(evaluation.artifacts["soup"].get_text())
        coverage_ratio = (
            coverage / text_length
            if text_length > 0
            else 0
        )
        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Total annotated text ratio: {coverage} characters.",
            score=coverage_ratio,
            details={"coverage_characters": coverage, "total_characters": text_length},
        )
        return scoring
