from typing import Any, List, Type

from argdown_feedback.api.shared.filtering import FilterRoleType

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

from .....shared.models import ScoringResult
from ..base import BaseScorer, VerifierBuilder
from nltk.tokenize import sent_tokenize  # type: ignore


### Scorers ###

class AnnotationCoverageScorer(BaseScorer):
    """Scorer that evaluates the coverage of argumentative annotations in the input."""

    scorer_id = "annotation_coverage_scorer"
    scorer_description = "Scores the text coverage of the annotation, i.e. the ratio of annotated text to total text length."

    def score(self, result: VerificationRequest) -> ScoringResult:

        soup, _ = self.get_xml_soup(result)
        if not soup:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No XML content found for annotation coverage scoring.",
                score=0.0,
                details={},
            )        
        propositions = soup.find_all("proposition")
        coverage = sum(len(proposition.get_text()) for proposition in propositions)
        text_length = len(soup.get_text())
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


class AnnotationScopeScorer(BaseScorer):
    """Scorer that evaluates the number of annotated proposition elements."""

    scorer_id = "annotation_scope_scorer"
    scorer_description = "Number of annotated proposition elements."

    def score(self, result: VerificationRequest) -> ScoringResult:
        soup, _ = self.get_xml_soup(result)

        if not soup:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No XML content found for annotation scope scoring.",
                score=0.0,
                details={},
            )

        text = soup.get_text() if soup else ""
        # Tokenize the text into sentences using NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        propositions = soup.find_all("proposition")
        score = len(propositions)/len(sentences) if sentences else 0
        score = min(score, 1.0)  # Cap the score at 1.0
        return ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Number of annotated propositions: {score}.",
            score=score,
            details={"annotated_proposition_count": len(propositions), "all_sentences_count": len(sentences)},
        )


class AnnotationDensityScorer(BaseScorer):
    """Scorer that evaluates the number of support and attack relations between propositions."""

    scorer_id = "annotation_density_scorer"
    scorer_description = "Scores based on the number of dialectic relations between propositions."

    def score(self, result: VerificationRequest) -> ScoringResult:
        soup, _ = self.get_xml_soup(result)
        propositions = soup.find_all("proposition") if soup is not None else None

        if not soup or not propositions:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No XML content found for annotation supports scoring.",
                score=0.0,
                details={},
            )

        propositions_count = len(propositions)
        supports_count = sum(len(proposition.get("supports", [])) for proposition in propositions) # type: ignore
        attacks_count = sum(len(proposition.get("attacks", [])) for proposition in propositions) # type: ignore
        score = (supports_count + attacks_count) / (2 * propositions_count) if propositions_count > 0 else 0
        score = min(score, 1.0)  # Cap the score at 1.0

        return ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Number of dialectic relations identified per proposition: {(supports_count + attacks_count) / (propositions_count) if propositions_count else '--'}.",
            score=score,
            details={"support_count": supports_count, "attack_count": attacks_count, "proposition_count": propositions_count},
        )


### Verifier builder ###

class ArgannoBuilder(VerifierBuilder):
    """Builder for argumentative annotation verifier."""

    name = "arganno"
    description = "Validates argumentative annotations in XML format"
    input_types = ["xml"]
    allowed_filter_roles = ["arganno"]
    scorer_classes: List[Type[BaseScorer]] = [
        AnnotationCoverageScorer,
        AnnotationScopeScorer,
        AnnotationDensityScorer,
    ]
    config_options = []  
    
    def build_handlers_pipeline(
        self, filters_spec: dict[FilterRoleType, Any], **kwargs
    ) -> List[BaseHandler]:
        """Build arganno verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            XMLParser(name="XMLAnnotationParser"),
            HasAnnotationsHandler(filter=vd_filters.get("arganno")),
            ArgannoCompositeHandler(filter=vd_filters.get("arganno")),
        ]
    
