from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
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
from argdown_feedback.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    Annotation,
    AnnotationProblem,
)
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap,
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
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.coherence.arganno_argmap_handler import (
    ArgannoArgmapCoherenceHandler,
)


class ArgmapPlusArgannoProblem(ArgMapProblem, AnnotationProblem):
    """Task: Create coherent argmap and arg annotation."""

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
            # Assignment: Annotate a source text and reconstruct its argumentation as an Argdown argument map.
                        
            Analyse the argumentation in the following **source text**. Your answer is supposed to contain 
            1. an argumentative text annotation and, in addition,
            2. a separate Argdown argument map.
                   
            Both, annotation and Argdown argument map, must cohere with each other.

            ::: {{.source_text}}
            {sources}
            :::

            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                   
            ## Argument Mapping Task Details                   

            Create a syntactically correct Argdown argument map that represents the overall argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

            Importantly, enclose your Argdown argument map in a separate fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation and Argument Map

            The argument map and the annotated source text must cohere with each other. There should be a one-to-many correspondence between argument map nodes and annotated text segments. Moreover, the support and attack relations in the argument map should reflect the annotated dialectical relations.
                   
            In particular, you should ensure that: 

            - Every <proposition> element in the annotation has an `argument_label` attribute that refers to a node (label of claim or argument) in the argument map.
            - Every node in the Argdown argument map has yaml inline data with an `annotation_ids` attribute that contains a list of `id` attributes of the corresponding <proposition> element in the annotation.
            - Two nodes in the argument map support each other if and only if the corresponding <proposition> elements are annotated to support each other (`support` attribute).
            - Two nodes in the argument map attack each other if and only if the corresponding <proposition> elements are annotated to attack each other (`support` attribute).
                   
            ## Output Format
                   
            Your answer must contain at least two fenced codeblocks: one for the annotated source text and one for the Argdown argument map. For example:
                   
            ```xml
            // Annotated source text here
            ``` 
                   
            ```argdown
            // Argdown argument map here
            ```
                   
            Don't forget the three closing backticks for the fenced codeblocks!

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
        prompt = "Revise your previously submitted annotation and argument map given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgmapPlusArganno(Annotation, ArgumentMap):
    """
    Solution to the ArgmapPlusArganno problem: annotation and argdown snippet.

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
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusArganno":
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


class ArgmapPlusArgannoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusArgannoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + argument mapping problem must be a string or a list of strings"
        )


class ArgmapPlusArgannoJudge(MPJudge):
    """Judge for the anno plus argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgmapPlusArgannoProblem), (
            "Problem must be an ArgmapPlusArgannoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusArganno)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgmapPlusArganno) for solution in solutions
        ), "All solutions must be ArgmapPlusArganno objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgmapPlusArgannoProblem), "Problem must be an ArgmapPlusArgannoProblem"
        assert isinstance(solution, ArgmapPlusArganno), "Solution must be an ArgmapPlusArganno"

        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasAnnotationsHandler(),
                HasArgdownHandler(),
                ArgannoCompositeHandler(),
                ArgMapCompositeHandler(),
                ArgannoArgmapCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation




class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid solutions
    where the source text's annotated propositions are textually similiar to the node texts in the argument map."""

    hints = [
        "Make sure that your argument map stays faithful to and mimics closely "
        "the annotation of the source text. In particular, use a similar wording for claims as "
        "in the corresponding annotated source segments!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        soup = evaluation.artifacts.get("soup")
        argdown = evaluation.artifacts.get("argdown_map")
        assert soup and argdown, (
            "AnnotationProximityPreferencePairGenerator: Missing soup or argdown in evaluation artifacts"
        )
        assert isinstance(soup, BeautifulSoup), "soup must be a BeautifulSoup object"
        assert isinstance(argdown, ArgdownMultiDiGraph), (
            "argdown must be an ArgdownMultiDiGraph object"
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

        if not dlss:
            return 0.0

        return round(sum(dlss) / len(dlss), 1)
