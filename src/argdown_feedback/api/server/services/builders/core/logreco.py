from typing import List

from nltk.sem.logic import Expression  # type: ignore
from pyargdown import ArgdownMultiDiGraph, Conclusion
import textdistance

from argdown_feedback.api.server.services.verifier_registry import BaseScorer
from argdown_feedback.logic.fol_to_nl import FOL2NLTranslator
from argdown_feedback.logic.logic import get_propositional_variables
from argdown_feedback.tasks.base import Evaluation
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
from argdown_feedback.verifiers.verification_request import VerificationRequest

from ..base import VerifierBuilder
from .....shared.models import ScoringResult, VerifierConfigOption

from .infreco import InfrecoPremisesScorer, InfrecoSubargumentsScorer, InfrecoFaithfulnessScorer


### Scorers ###

class LogrecoPremisesScorer(InfrecoPremisesScorer):
    """Scorer for premises in logical argument reconstruction."""
    
    scorer_id = "logreco_premises_scorer"
    scorer_description = "Scores the number of premises in the logical argument reconstruction."


class LogrecoSubargumentsScorer(InfrecoSubargumentsScorer):
    """Scorer for sub-arguments in logical argument reconstruction."""
    
    scorer_id = "logreco_subarguments_scorer"
    scorer_description = "Scores the number of sub-arguments in the logical argument reconstruction."


class LogrecoFaithfulnessScorer(InfrecoFaithfulnessScorer):
    """Scorer for faithfulness of logical argument reconstruction to the input text."""
    
    scorer_id = "logreco_faithfulness_scorer"
    scorer_description = "Scores the faithfulness of the logical argument reconstruction to the input text."


class LogrecoPredicateLogicScorer(BaseScorer):
    """Scorer that evaluates the use of predicate logic in the logical argument reconstruction."""

    scorer_id = "logreco_predicate_logic_scorer"
    scorer_description = "Scores the use of predicate logic in the logical argument reconstruction."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        all_expressions = evaluation.artifacts.get("all_expressions")
        if not all_expressions:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No logical expressions found; cannot compute predicate logic score.",
                score=0.0,
                details={},
            )
        
        n_has_prop_vars = sum(bool(get_propositional_variables(expr)) for expr in all_expressions.values())
        score = 1 - (n_has_prop_vars / len(all_expressions))
        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Found {n_has_prop_vars} out of {len(all_expressions)} expressions using propositional variables.",
            score=score,
            details={
                "expressions_with_propositional_variables": n_has_prop_vars,
                "total_expressions": len(all_expressions),
            },
        )
        return scoring


class LogrecoFormalizationsFaithfulnessScorer(BaseScorer):
    """Scores faithfulness of ofmalizations argument reco, rewarding valid reconstructions
    with formalizations that are similiar to the sentences being formalized."""

    scorer_id = "logreco_formalizations_faithfulness_scorer"
    scorer_description = "Scores the faithfulness of formalizations in the logical argument reconstruction to the input text."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown")
        all_expressions = evaluation.artifacts.get("all_expressions")
        all_declarations = evaluation.artifacts.get("all_declarations")

        if not argdown or not all_declarations or not all_expressions or not argdown.arguments:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No formalizations or argument found; cannot compute formalizations faithfulness score.",
                score=0.0,
                details={},
            )

        argument = argdown.arguments[0]

        dlds: list[float] = []
        for pr in argument.pcs:
            expression = all_expressions.get(pr.proposition_label)
            proposition = argdown.get_proposition(pr.proposition_label)

            if expression is None or proposition is None:
                continue 

            text_1 = FOL2NLTranslator.translate_to_nl_sentence(
                expression, all_declarations
            )

            for text_2 in proposition.texts:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_similarity(
                        text_1, text_2
                    )
                )

        score = round(sum(dlds) / len(dlds), 1) if dlds else 0

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Average faithfulness score of formalizations to sentences: {score:.2f}.",
            score=score,
            details={"faithfulness_per_proposition": dlds},
        )
        return scoring

class LogrecoTrivialityScorer(BaseScorer):
    """Scores that the FOL inference from premises to final conclusion is not trivial, i.e.
    does in particular not just consists in joining the premises via conjunction as conclusion."""

    scorer_id = "logreco_triviality_scorer"
    scorer_description = "Scores the degree to which the reconstructed inference is not trivial."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown")
        all_expressions: dict[str,Expression] | None = evaluation.artifacts.get("all_expressions")
        if not all_expressions or not argdown or not argdown.arguments:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No logical expressions / argument found; cannot compute triviality score.",
                score=0.0,
                details={},
            )
        argument = argdown.arguments[0]

        premises_formalized = [
            all_expressions[pr.proposition_label]
            for pr in argument.pcs
            if pr.proposition_label in all_expressions and not isinstance(pr, Conclusion)
        ]
        conclusion = next(
            all_expressions[pr.proposition_label]
            for pr in reversed(argument.pcs)
            if pr.proposition_label in all_expressions and isinstance(pr, Conclusion)
        )

        # Hacky string-based check for triviality
        conclusion_conjuncts = str(conclusion).split('&')
        conclusion_conjuncts = [c.strip() for c in conclusion_conjuncts]
        premises_stripped = [str(p).strip() for p in premises_formalized]
        trivial = all(conjunct in premises_stripped for conjunct in conclusion_conjuncts)

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"The final conclusion is {'trivially' if trivial else 'non-trivially'} derived from the premises.",
            score=0.0 if trivial else 1.0,
            details={},
        )
        return scoring


### Verifier Builder ###

class LogrecoBuilder(VerifierBuilder):
    """Builder for logical argument reconstruction verifier."""
    
    name = "logreco"
    description = "Validates logical argument reconstruction in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["logreco"]
    scorer_classes = [
        LogrecoPremisesScorer,
        LogrecoSubargumentsScorer,
        LogrecoFaithfulnessScorer,
        LogrecoPredicateLogicScorer,
        LogrecoFormalizationsFaithfulnessScorer,
        LogrecoTrivialityScorer
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

