from typing import List

from pyargdown import ArgdownMultiDiGraph, Conclusion
import textdistance

from argdown_feedback.api.server.services.verifier_registry import BaseScorer
from argdown_feedback.tasks.base import Evaluation
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
from argdown_feedback.verifiers.verification_request import VerificationDType, VerificationRequest

from ..base import VerifierBuilder
from .....shared.models import ScoringResult, VerifierConfigOption


### Scorers ###


class InfrecoSubargumentsScorer(BaseScorer):
    """Scorer that evaluates the number of sub-arguments in the informal reconstruction."""

    scorer_id = "infreco_subarguments_scorer"
    scorer_description = "Scores the number of sub-arguments in the informal reconstruction."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown")

        if not argdown or not argdown.arguments:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No argument reconstruction found; cannot compute sub-arguments score.",
                score=0.0,
                details={},
            )
        
        argument = argdown.arguments[0]
        conclusion_count = sum(
            1 for prop in argument.pcs if isinstance(prop, Conclusion)
        )
        score = 1 - 0.5 ** conclusion_count

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Number of sub-arguments (intermediate conclusions) found: {conclusion_count-1}.",
            score=score,
            details={"intermediate_conclusion_count": conclusion_count - 1},
        )

        return scoring


class InfrecoPremisesScorer(BaseScorer):
    """Scorer that evaluates the number of premises in the informal reconstruction."""

    scorer_id = "infreco_premises_scorer"
    scorer_description = "Scores the number of premises in the informal reconstruction."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown")

        if not argdown or not argdown.arguments:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No argument reconstruction found; cannot compute premises score.",
                score=0.0,
                details={},
            )
        
        argument = argdown.arguments[0]
        premises_count = sum(
            1 for prop in argument.pcs if not isinstance(prop, Conclusion)
        )
        score = 1 - 0.7 ** max(premises_count-1, 0)

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Number of premises found: {premises_count}.",
            score=score,
            details={"premises_count": premises_count - 1},
        )

        return scoring


class InfrecoFaithfulnessScorer(BaseScorer):
    """Scorer that evaluates the faithfulness of the informal argument reconstruction to the input."""

    scorer_id = "infreco_faithfulness_scorer"
    scorer_description = "Scores the faithfulness of the informal argument reconstruction, i.e. the text similarity between argdown snippet and source text."

    def score(self, result: VerificationRequest) -> ScoringResult:

        source_text = result.source
        argdown_snippet = next(
            (
                vr.code_snippet for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )

        if not source_text or not argdown_snippet:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No source text / argdown provided; cannot compute faithfulness score.",
                score=0.0,
                details={},
            )

        text_similarity = round(
            textdistance.damerau_levenshtein.normalized_similarity(
                source_text, argdown_snippet
            ),
            1,
        )

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Text similarity between argdown snippet and source text: {text_similarity}.",
            score=text_similarity,
            details={},
        )
        return scoring


### Verifier Builder ###

class InfrecoBuilder(VerifierBuilder):
    """Builder for informal argument reconstruction verifier."""
    
    name = "infreco"
    description = "Validates informal argument reconstruction in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["infreco"]
    scorer_classes = [
        InfrecoSubargumentsScorer,
        InfrecoPremisesScorer,
        InfrecoFaithfulnessScorer
    ]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False
        ),
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


