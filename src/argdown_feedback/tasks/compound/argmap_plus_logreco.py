import dataclasses
from typing import Sequence

from textwrap import dedent
from pyargdown import (
    ArgdownMultiDiGraph,
)
import textdistance

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_feedback.logic.fol_to_nl import FOL2NLTranslator
from argdown_feedback.tasks.core.logreco import (
    LogicalReco,
)
from argdown_feedback.tasks.compound.argmap_plus_infreco import (
    ArgmapPlusInfreco,
    ArgmapPlusInfrecoProblem,
)
from argdown_feedback.verifiers.base import BaseHandler, CompositeHandler
from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    ArgmapLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_feedback.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasAtLeastNArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    NoExtraPropositionsHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
)


class ArgmapPlusLogrecoProblem(ArgmapPlusInfrecoProblem):
    """Task: Create coherent logical reco and argument map."""

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
            # Assignment: Present a text's argumentation as an informal Argdown argument map, and logically reconstruct its arguments in standard form using Argdown syntax.

            Analyse the argumentation in the given **source text**. Your answer is supposed to contain two artifacts:
            1. an Argdown argument map and
            2. an Argdown snippet with logical reconstructions of all the arguments in standard form (as deductively valid inferences).

            In the following, you find
            * the source text to analyse,
            * detailed instructions for how to create the Argdown argument map (first artifact),
            * detailed instructions for how to logically reconstruct and formalize the arguments (second artifact),
            * a description of how both artifacts are supposed to cohere with each other,
            * formatting instructions for your answer.
                        
            ## Source Text
                   
            ::: {{.source_text}}
            {sources}
            :::
                   
            ## Argument Mapping Task Details                   
                   
            Create a syntactically correct Argdown argument map that captures the overall argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
            - cover *at least two* arguments in the argument map;

            Importantly, enclose your Argdown argument map in a fenced codeblock:
            ```argdown {{filename="map.ad"}}
            // your Argdown argument map here
            ```
            If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Argument Reconstruction Task Details                   

            Logically analyse and reconstruct the text's arguments with Argdown, ensuring the inferences are deductively valid.
            - Reconstruct all arguments presented in the map in standard form (including argument title, premises, final conclusion, and possible intermediate conclusions).      
            - For each premise and conclusion in your reconstructions, provide an adequate propositional logic / FOL formalization in NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Only declare variables that are used in the corresponding formalization and that have not been declared in the corresponding argument before. Ensure that your formalizations are consistent across different arguments.
            - For each inference step in the argument, provide information about which previously introduced premises or conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.
            - Use `<-` / `<+` / `><` syntax to declare that any premises and/or conclusions from different arguments logically entail or contradict each other, providing explicit labels for these claims in square brackets.

            Importantly, enclose your Argdown reconstructions in a single fenced codeblock, separating different arguments with newlines:
            ```argdown {{filename="reconstructions.ad"}}'
            // your formal Argdown reconstructions here
            ```
            If you provide multiple Argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation and Argument Reconstruction

            The argument map and your argument reconstructions must neatly correspond to each other. Meaning that:

            1. Every argument in the _argument map_ is reconstructed in standard form.
            2. Every reconstructed argument is present in the _argument map_.
            3. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) figures as premise in the reconstructed argument -- and vice versa.
            4. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) figures as conclusion in the reconstructed argument -- and vice versa.
            5. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) figures as premise of B -- and vice versa.
            6. Whenever a claim A, in the _argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
                   
            Here are the specific notation instructions which help you to ensure that your argument map, on the one hand, and your argument reconstructions, on the other hand, fully cohere with each other in the above sense: 

            - The argument labels in the argument map (angle brackets) must match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map (square brackets) for the corresponding premises and conclusions (if any) in the argument reconstruction.
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax).
            """)
            .strip()
            .format(sources=self.sources)
        )

        if hints:
            prompt += "\n\n## Hints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

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
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgmapPlusLogreco(ArgmapPlusInfreco):
    """
    Solution to the ArgmapPlusLogreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    def partial_logreco(self) -> LogicalReco:
        """Return the informal reconstruction subsolution."""
        return LogicalReco(
            argdown_snippet=self.argdown_reconstructions_snippet, _raw_answer=self._raw_answer,
        )


class ArgmapPlusLogrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusLogrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgmapPlusLogrecoJudge(MPJudge):
    """Judge for the argmap plus infreco task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgmapPlusLogrecoProblem), (
            "Problem must be an ArgmapPlusLogrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusLogreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgmapPlusLogreco) for solution in solutions
        ), "All solutions must be ArgmapPlusLogreco objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgmapPlusLogrecoProblem), "Problem must be an ArgmapPlusLogrecoProblem"
        assert isinstance(solution, ArgmapPlusLogreco), "Solution must be an ArgmapPlusLogreco"

        map_filter = BaseHandler.create_metadata_filter("filename", ["map.ad"])
        reco_filter = BaseHandler.create_metadata_filter(
            "filename", ["reconstructions.ad"]
        )

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasAtLeastNArgumentsHandler(name="InfReco.HasAtLeastNArgumentsHandler",filter=reco_filter, N=2),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=reco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=reco_filter),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=reco_filter),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=reco_filter),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=reco_filter),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", filter=reco_filter),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", filter=reco_filter),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", filter=reco_filter),
                # Extra material handlers
                NoExtraPropositionsHandler(name="InfReco.NoExtraPropositionsHandler", filter=reco_filter),
            ]
        )
        main_handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                ArgdownParser(name="ArgdownParser"),
                HasArgdownHandler(name="HasArgdownHandler.map", filter=map_filter),
                HasArgdownHandler(name="HasArgdownHandler.reco", filter=reco_filter),
                ArgMapCompositeHandler(filter=map_filter),
                infreco_handler,
                LogRecoCompositeHandler(filter=reco_filter),
                ArgmapInfrecoCoherenceHandler(),
                ArgmapLogrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = main_handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation


# class ArgmapPlusLogrecoJudge2(Judge):
#     """Judge for the argmap plus infreco task."""

#     def _evaluate_solution(
#         self, problem: ArgmapPlusLogrecoProblem, solution: ArgmapPlusLogreco
#     ) -> Evaluation:
#         map_filter = BaseHandler.create_metadata_filter("filename", ["map.ad"])
#         reco_filter = BaseHandler.create_metadata_filter(
#             "filename", ["reconstructions.ad"]
#         )

#         infreco_handler = InfRecoCompositeHandler(
#             handlers=[
#                 # Argument existence handlers
#                 HasAtLeastNArgumentsHandler(name="InfReco.HasAtLeastNArgumentsHandler",filter=reco_filter, N=2),
#                 HasPCSHandler(name="InfReco.HasPCSHandler", filter=reco_filter),
#                 # Argument form handlers
#                 StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=reco_filter),
#                 EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=reco_filter),
#                 NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=reco_filter),
#                 # Label and gist handlers
#                 HasLabelHandler(name="InfReco.HasLabelHandler", filter=reco_filter),
#                 # Inference data handlers
#                 HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", filter=reco_filter),
#                 PropRefsExistHandler(name="InfReco.PropRefsExistHandler", filter=reco_filter),
#                 UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", filter=reco_filter),
#                 # Extra material handlers
#                 NoExtraPropositionsHandler(name="InfReco.NoExtraPropositionsHandler", filter=reco_filter),
#             ]
#         )
#         main_handler = CompositeHandler(
#             handlers=[
#                 DefaultProcessingHandler(),
#                 HasArgdownHandler(name="HasArgdownHandler.map", filter=map_filter),
#                 HasArgdownHandler(name="HasArgdownHandler.reco", filter=reco_filter),
#                 ArgMapCompositeHandler(filter=map_filter),
#                 infreco_handler,
#                 LogRecoCompositeHandler(filter=reco_filter),
#                 ArgmapInfrecoCoherenceHandler(),
#                 ArgmapLogrecoCoherenceHandler(),
#             ]
#         )
#         request = VerificationRequest(inputs=str(solution), source=problem.sources)
#         result = main_handler.process(request)
#         evaluation = Evaluation.from_verification_request(result)
#         return evaluation

#     async def arun(
#         self,
#         problem: Problem,
#         solutions: Sequence[Solution],
#         original_solution: Solution | None = None,
#         feedback: Feedback | None = None,
#     ) -> Sequence[Evaluation]:
#         assert isinstance(problem, ArgmapPlusLogrecoProblem), (
#             "Problem must be an ArgannoPlusLogRecoProblem"
#         )
#         assert (
#             isinstance(original_solution, ArgmapPlusLogreco)
#             or original_solution is None
#         )
#         assert feedback or original_solution is None, (
#             "Feedback is required for evaluating revised solutions"
#         )

#         evaluations = []
#         for solution in solutions:
#             assert isinstance(solution, ArgmapPlusLogreco), (
#                 "All solutions must be ArgmapPlusLogreco"
#             )
#             evaluations.append(self._evaluate_solution(problem, solution))

#         return evaluations


class GlobalFormalizationsFaithfulnessPreferencePairGenerator(
    ScoringVirtuePreferencePairGenerator
):
    """Global FormalizationsFaithfulnessPreferencePairGenerator"""

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown_reco = evaluation.artifacts.get("argdown_reco")
        assert argdown_reco is not None and isinstance(
            argdown_reco, ArgdownMultiDiGraph
        ), "Evaluation must contain argdown_reco artifact"
        all_expressions = evaluation.artifacts.get("all_expressions")
        assert all_expressions is not None and isinstance(all_expressions, dict), (
            "Evaluation must contain all_expressions artifact"
        )
        all_declarations = evaluation.artifacts.get("all_declarations")
        assert all_declarations is not None and isinstance(all_declarations, dict), (
            "Evaluation must contain all_declarations artifact"
        )

        dlds: list[float] = []
        for argument in argdown_reco.arguments:
            #print(f"Argument: {argument.label}")
            for pr in argument.pcs:
                expression = all_expressions.get(pr.proposition_label)
                if expression is None:
                    continue

                proposition = argdown_reco.get_proposition(pr.proposition_label)
                if proposition is None:
                    continue

                text_1 = FOL2NLTranslator.translate_to_nl_sentence(
                    expression, all_declarations
                )
                #print(f"Text 1: {text_1}")

                for text_2 in proposition.texts:
                    #print(f"Text 2: {text_2}")
                    dlds.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text_1, text_2
                        )
                    )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0
