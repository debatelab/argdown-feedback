import dataclasses
from textwrap import dedent
from typing import Sequence

from pyargdown import ArgdownMultiDiGraph, Proposition

from argdown_feedback.tasks.base import (
    Evaluation,
    Feedback,
    Judge,
    Problem,
    ProblemGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap,
    ConnectednessPreferencePairGenerator,
    MaxArgsPreferencePairGenerator,
    MaxAttacksPreferencePairGenerator,
    MaxSupportsPreferencePairGenerator,
    SourceTextProximityPreferencePairGenerator,
)
from argdown_feedback.tasks.core.infreco import InfRecoProblem, InformalReco
from argdown_feedback.verifiers.base import BaseHandler, CompositeHandler
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
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
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)


class ArgmapPlusInfrecoProblem(InfRecoProblem, ArgMapProblem):
    """Task: Create coherent informal reco and argument map."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            # Assignment: Present a text's argumentation as an informal Argdown argument map, and reconstruct its arguments in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create two coherent Argdown code snippets: One with an informal argument map, and another one with reconstructions of all the arguments in standard form (premise-conclusion structure).

            ::: {{.source_text}}              
            {sources}
            :::

                   
            ## Argument Mapping Task Details                   
                   
            Create a syntactically correct informal Argdown argument map that reconstructs the argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

            Importantly, enclose your Argdown argument map in a fenced codeblock, starting with '```argdown {{filename="map.ad"}}' and ending with '```'. If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.


            ## Argument Reconstruction Task Details                   

            Informally analyse and reconstruct the text's arguments with Argdown. In particular, you should

            - reconstruct *at least two arguments* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).
            - provide, for each conclusion in an argument, information about which previously introduced premises or 
              conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
            - ensure that every premise and intermdeiate conclusions is actually used to infer a conclusion in the argument.
                  
            Importantly, enclose your Argdown reconstructions in a fenced codeblock, starting with '```argdown {{filename="reconstructions.ad"}}' and ending with '```'. If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

                   
            ## Required Coherence of Annotation and Argument Reconstruction                                            

            The argument map and your argument reconstructions must neatly correspond to each other. Meaning that:
                   
            1. Every argument in the argument map is reconstructed in standard form.
            2. Every reconstructed argument is present in the argument map.
            3. Whenever a claim in the argument map supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument -- and vice versa.
            4. Whenever an argument in the argument map supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument -- and vice versa.
            5. Whenever an argument A in the argument map supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa.
                   
            Here are the specific notation instructions which help you to ensure that argument map and argument reconstructions fully cohere with each other in the above sense: 

            - The argument labels in the argument map must match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstruction. 
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label or, absent any label, have string-identical texts.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if they have different labels, and one text prepends "NOT: " the other text. (Avoid double negations and rely on duplex negatio affirmat instead.)
        """)
            .strip()
            .format(sources=self.sources)
        )

        if hints:
            prompt += "\n\n## Hints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
                "\n\n"
                "> [!WARNING]\n"
                "> For didactic purposes, I want you to make mistakes in your answer.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"

        return prompt

    def revise_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = "Revise your previously submitted argument map and argument reconstructions given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
                "\n\n"
                "> [!WARNING]\n"
                "> For didactic purposes, I still want you to make mistakes in your revised answer.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"

        return prompt


@dataclasses.dataclass
class ArgmapPlusInfreco(Solution):
    """
    Solution to the ArgmapPlusInfreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    argdown_map_snippet: str
    argdown_reconstructions_snippet: str
    unparsed_solution: str | None = None

    def __str__(self):
        if self.unparsed_solution:
            return self.unparsed_solution
        return self.argdown_map_snippet + "\n\n" + self.argdown_reconstructions_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusInfreco":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.handle(request)

        map_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown
                and vr.code_snippet
                and vr.metadata
                and vr.metadata.get("filename") == "map.ad"
            ),
            None,
        )
        reco_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown
                and vr.code_snippet
                and vr.metadata
                and vr.metadata.get("filename") == "reconstructions.ad"
            ),
            None,
        )

        return cls(
            argdown_map_snippet=map_snippet if map_snippet else raw_answer,
            argdown_reconstructions_snippet=reco_snippet
            if reco_snippet
            else raw_answer,
            unparsed_solution=None if map_snippet and reco_snippet else raw_answer,
        )

    def partial_argmap(self) -> ArgumentMap:
        """Return the argument map subsolution."""
        return ArgumentMap(
            argdown_snippet=self.argdown_map_snippet,
        )

    def partial_infreco(self) -> InformalReco:
        """Return the informal reconstruction subsolution."""
        return InformalReco(
            argdown_snippet=self.argdown_reconstructions_snippet,
        )


class ArgmapPlusInfrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusInfrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgmapPlusInfrecoJudge(Judge):
    """Judge for the argmap plus infreco task."""

    def _evaluate_solution(
        self, problem: ArgmapPlusInfrecoProblem, solution: ArgmapPlusInfreco
    ) -> Evaluation:
        map_filter = BaseHandler.create_metadata_filter("filename", ["map.ad"])
        reco_filter = BaseHandler.create_metadata_filter(
            "filename", ["reconstructions.ad"]
        )

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(filter=reco_filter),
                HasPCSHandler(filter=reco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(filter=reco_filter),
                EndsWithConclusionHandler(filter=reco_filter),
                NoDuplicatePCSLabelsHandler(filter=reco_filter),
                # Label and gist handlers
                HasLabelHandler(filter=reco_filter),
                # Inference data handlers
                HasInferenceDataHandler(filter=reco_filter),
                PropRefsExistHandler(filter=reco_filter),
                UsesAllPropsHandler(filter=reco_filter),
            ]
        )
        main_handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasArgdownHandler(filter=map_filter),
                HasArgdownHandler(filter=reco_filter),
                ArgMapCompositeHandler(filter=map_filter),
                infreco_handler,
                ArgmapInfrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = main_handler.handle(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgmapPlusInfrecoProblem), (
            "Problem must be an ArgannoPlusInfrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusInfreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgmapPlusInfreco), (
                "All solutions must be ArgmapPlusInfreco"
            )
            evaluations.append(self._evaluate_solution(problem, solution))

        return evaluations


class SimplicityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the ArgmapPlusInfreco, prefering valid reconstructions
    with succinct and simple propositions."""

    hints = [
        "Make sure that you keep each of the arguments premises and conclusion(s) simple and succinct. "
        "Short sentences are crucial at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_reco" in evaluation.artifacts, (
            "Evaluation must contain argdown_reco artifact"
        )
        argdown_reco: ArgdownMultiDiGraph = evaluation.artifacts["argdown_reco"]
        propositions: list[Proposition] = argdown_reco.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) ** -1 if lengths else 0


class ConnectednessPreferencePairGeneratorCT(ConnectednessPreferencePairGenerator):
    """Simple wrapper around ConnectednessPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )


class MaxArgsPreferencePairGeneratorCT(MaxArgsPreferencePairGenerator):
    """Simple wrapper around MaxArgsPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide partial argmap"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )


class MaxSupportsPreferencePairGeneratorCT(MaxSupportsPreferencePairGenerator):
    """Simple wrapper around MaxSupportsPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )


class MaxAttacksPreferencePairGeneratorCT(MaxAttacksPreferencePairGenerator):
    """Simple wrapper around MaxAttacksPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )


class SourceTextProximityPreferencePairGeneratorCT(
    SourceTextProximityPreferencePairGenerator
):
    """Simple wrapper around SourceTextProximityPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )
