from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import (
    Conclusion,
    Argument,
)
import textdistance

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_feedback.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    Annotation,
    AnnotationProblem,
)
from argdown_feedback.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco,
)
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
    HasAnnotationsHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
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
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)


# utility #


def _get_props_used_in_inference(
    argument: Argument, pr_label: str, from_key: str = "from"
) -> list[str]:
    """Get all proposition labels used directly or indirectly in the inference
    to a conclusion with label `pr_label`."""

    if argument is None or not argument.pcs:
        return []

    used_labels = set()

    def add_parent_labels(label: str):
        c = next(
            (c for c in argument.pcs if isinstance(c, Conclusion) and c.label == label),
            None,
        )
        if c is None:
            return []
        parent_labels = c.inference_data.get(from_key, [])
        used_labels.update(parent_labels)
        for ref in parent_labels:
            add_parent_labels(ref)

    add_parent_labels(pr_label)

    return list(used_labels)


class ArgannoPlusInfrecoProblem(InfRecoProblem, AnnotationProblem):
    """Task: Create coherent informal reco and arg annotation."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        sources = BeautifulSoup(sources, "html.parser").get_text()
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
            # Assignment: Annotate a source text and reconstruct its main argument in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create a a coherent argumentative text annotation and a corresponding informal argument reconstruction in standard form (premise-conclusion structure).

            ::: {{.source_text}}              
            {sources}
            :::

            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                   
            ## Argument Reconstruction Task Details                   

            Informally analyse and reconstruct the text's main argumentation with Argdown. In particular, you should

            - reconstruct *at least one argument* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).
            - provide, for each conclusion in an argument, information about which previously introduced premises or 
              conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
                  
            Importantly, enclose your Argdown snippet in a fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation and Argument Reconstruction                                                

            The argument reconstruction and the annotated source text must cohere with each other. There should be a one-to-many correspondence between premises/conclusion(s) and annotated text segments. Moreover, the inferential relations in the reconstructed argument should reflect the annotated support relations.
                   
            In particular, you should ensure that: 

            - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippet.
            - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding argument. 
            - Every premise and conclusion in the Argdown argument has yaml inline data with an `annotation_ids` attribute that contains a list of `id` attributes of the corresponding <proposition> elements in the annotation.
            - If, in the annotation, one <proposition> element supports another one (via its `support` attribute), then, in the Argdown argument, the proposition corresponding to the former element is used to infer the conclusion corresponding to the latter element.
        """)
            .strip()
            .format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)
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
        prompt = "Revise your previously submitted annotation and argument reconstruction given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgannoPlusInfreco(Annotation, InformalReco):
    """
    Solution to the ArgannoPlusInfreco problem: annotation and argdown snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    annotated_source_text: str
    argdown_snippet: str
    unparsed_solution: str | None = None

    def __str__(self):
        if self.unparsed_solution:
            return self.unparsed_solution
        return self.annotated_source_text + "\n\n" + self.argdown_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgannoPlusInfreco":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)

        annotated_source_text = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml and vr.code_snippet
            ),
            None,
        )
        argdown_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )

        return cls(
            annotated_source_text=annotated_source_text
            if annotated_source_text
            else raw_answer,
            argdown_snippet=argdown_snippet if argdown_snippet else raw_answer,
            unparsed_solution=None
            if annotated_source_text and argdown_snippet
            else raw_answer,
        )


class ArgannoPlusInfrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgannoPlusInfrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgannoPlusInfrecoJudge(MPJudge):
    """Judge for the anno plus argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgannoPlusInfrecoProblem), (
            "Problem must be an ArgannoPlusInfrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgannoPlusInfreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgannoPlusInfreco) for solution in solutions
        ), "All solutions must be ArgannoPlusInfreco objects"


    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgannoPlusInfrecoProblem), "Problem must be an ArgannoPlusInfrecoProblem"
        assert isinstance(solution, ArgannoPlusInfreco), "Solution must be an ArgannoPlusInfreco"

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
            ]
        )

        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasAnnotationsHandler(),
                HasArgdownHandler(),
                ArgannoCompositeHandler(),
                infreco_handler,
                ArgannoInfrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation


# class ArgannoPlusInfrecoJudge2(Judge):
#     """Judge for the anno plus argument mapping task."""

#     def _evaluate_solution(
#         self, problem: ArgannoPlusInfrecoProblem, solution: Solution
#     ) -> Evaluation:
#         infreco_handler = InfRecoCompositeHandler(
#             handlers=[
#                 # Argument existence handlers
#                 HasArgumentsHandler(name="InfReco.HasArgumentsHandler"),
#                 HasPCSHandler(name="InfReco.HasPCSHandler"),
#                 # Argument form handlers
#                 StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler"),
#                 EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler"),
#                 NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler"),
#                 # Label and gist handlers
#                 HasLabelHandler(name="InfReco.HasLabelHandler"),
#                 # Inference data handlers
#                 HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler"),
#                 PropRefsExistHandler(name="InfReco.PropRefsExistHandler"),
#                 UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler"),
#             ]
#         )

#         handler = CompositeHandler(
#             handlers=[
#                 DefaultProcessingHandler(),
#                 HasAnnotationsHandler(),
#                 HasArgdownHandler(),
#                 ArgannoCompositeHandler(),
#                 infreco_handler,
#                 ArgannoInfrecoCoherenceHandler(),
#             ]
#         )
#         request = VerificationRequest(inputs=str(solution), source=problem.sources)
#         result = handler.process(request)
#         evaluation = Evaluation.from_verification_request(result)
#         if evaluation.artifacts.get("argdown_map") is None:
#             evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
#         return evaluation

#     async def arun(
#         self,
#         problem: Problem,
#         solutions: Sequence[Solution],
#         original_solution: Solution | None = None,
#         feedback: Feedback | None = None,
#     ) -> Sequence[Evaluation]:
#         assert isinstance(problem, ArgannoPlusInfrecoProblem), (
#             "Problem must be an ArgannoPlusInfrecoProblem"
#         )
#         assert (
#             isinstance(original_solution, ArgannoPlusInfreco)
#             or original_solution is None
#         )
#         assert feedback or original_solution is None, (
#             "Feedback is required for evaluating revised solutions"
#         )

#         evaluations = []
#         for solution in solutions:
#             assert isinstance(solution, ArgannoPlusInfreco), (
#                 "All solutions must be ArgannoPlusInfreco"
#             )
#             evaluations.append(self._evaluate_solution(problem, solution))

#         return evaluations



class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid solutions
    where the source text's annotated propositions are textually similiar to the propositions in the reconstructed argument."""
    
    hints = [
        (
            "Make sure that your argument reconstruction stays faithful to and mimics closely "
            "the annotation of the source text. In particular, use formulations of premises and conclusions "
            "that are similar to the corresponding annotated text segments!"
        )
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        soup = evaluation.artifacts["soup"]
        anno_props = soup.find_all("proposition")

        argdown = evaluation.artifacts["argdown_reco"]
        if argdown is None:
            argdown = evaluation.artifacts["argdown"]

        matches: list[tuple[str, str]] = []
        for proposition in argdown.propositions:
            for annotation_id in proposition.data.get("annotation_ids", []):
                anno_prop = next(
                    (ap for ap in anno_props if ap.get("id") == annotation_id), None
                )
                if anno_prop is None:
                    continue
                for text in proposition.texts:
                    matches.append((anno_prop.get_text(), text))

        #print("matches")
        #print(matches)
        dlss = [
            textdistance.damerau_levenshtein.normalized_similarity(s, t)
            for s, t in matches
        ]
        return round(sum(dlss) / len(dlss), 1)
