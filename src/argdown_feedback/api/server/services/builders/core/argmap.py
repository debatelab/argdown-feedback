from typing import List

import networkx as nx  # type: ignore
from pyargdown import Argdown, ArgdownMultiDiGraph
import textdistance

from argdown_feedback.api.server.services.verifier_registry import BaseScorer
from argdown_feedback.api.shared.models import ScoringResult, VerifierConfigOption
from argdown_feedback.tasks.base import Evaluation
from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import VerificationDType, VerificationRequest

from ..base import VerifierBuilder


class ArgmapBuilder(VerifierBuilder):
    """Builder for argument map verifier."""

    name = "argmap"
    description = "Validates argument maps in Argdown format"
    input_types = ["argdown"]
    allowed_filter_roles = ["argmap"]
    config_options = [
        VerifierConfigOption(
            name="enable_argmap_size_scorer",
            type="bool",
            default=False,
            description="Enable scoring of argument map size",
            required=False
        ),
        VerifierConfigOption(
            name="enable_argmap_density_scorer",
            type="bool",
            default=False,
            description="Enable scoring of argument map density",
            required=False
        ),
        VerifierConfigOption(
            name="enable_argmap_faithfulness_scorer",
            type="bool",
            default=False,
            description="Enable scoring of argument map faithfulness to input text",
            required=False
        ),
    ]

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build argmap verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            ArgdownParser(name="ArgdownParser"),
            HasArgdownHandler(filter=vd_filters.get("argmap")),
            ArgMapCompositeHandler(filter=vd_filters.get("argmap")),
        ]


### Scorers ###

class ArgmapSizeScorer(BaseScorer):
    """Scorer that evaluates the size of the argument map ."""

    scorer_id = "argmap_size_scorer"
    scorer_description = "Scores the size of the argument map, i.e. the number of arguments and theses reconstructed."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown_map")

        if not argdown:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No argument map found; cannot compute density score.",
                score=0.0,
                details={},
            )

        num_nodes = argdown.number_of_nodes()
        if num_nodes <= 3:
            score = 0.
        else:
            n = num_nodes - 3
            score = 1 - (0.6**n)

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Argument map size (number of nodes): {num_nodes}.",
            score=score,
            details={"number_of_nodes": num_nodes},
        )
        return scoring


class ArgmapDensityScorer(BaseScorer):
    """Scorer that evaluates the density of the argument map ."""

    scorer_id = "argmap_density_scorer"
    scorer_description = "Scores the desnity of the argument map, i.e. the number of interconnections between arguments and theses."

    def score(self, result: VerificationRequest) -> ScoringResult:
        evaluation = Evaluation.from_verification_request(result)
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown_map")

        if not argdown:
            return ScoringResult(
                scorer_id=self.name,
                scorer_description=self.scorer_description,
                scoring_data_references=[],
                message="No argument map found; cannot compute density score.",
                score=0.0,
                details={},
            )
        
        H = nx.DiGraph(argdown)
        degree_centrality = list(nx.degree_centrality(H).values())
        density = sum(degree_centrality) / len(degree_centrality) if degree_centrality else 0
        density = min(density, 1.0)  # Ensure the score does not exceed 1.0

        scoring = ScoringResult(
            scorer_id=self.name,
            scorer_description=self.scorer_description,
            scoring_data_references=[],
            message=f"Argument map density (average degree centrality): {density:.2f}.",
            score=density,
            details={"degree_centrality_per_node": degree_centrality},
        )
        return scoring


class ArgmapFaithfulnessScorer(BaseScorer):
    """Scorer that evaluates the faithfulness of the argument map to the input."""

    scorer_id = "argmap_faithfulness_scorer"
    scorer_description = "Scores the faithfulness of the argument map, i.e. the text similarity between argdown snippet and source text."

    def score(self, result: VerificationRequest) -> ScoringResult:

        # NOTE
        # This scorer assumes that the last argdown snippet is the target snippet (no filters applied)
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

