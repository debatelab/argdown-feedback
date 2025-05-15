from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup

from argdown_feedback.tasks.base import (
    Judge,
    MPJudge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
)
from argdown_feedback.tasks.compound.arganno_plus_infreco import (
    ArgannoPlusInfreco,
)
from argdown_feedback.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    AnnotationProblem,
)
from argdown_feedback.tasks.core.logreco import (
    LogRecoProblem,
)

from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
    HasAnnotationsHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)
from argdown_feedback.verifiers.verification_request import (
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
from argdown_feedback.verifiers.core.logreco_handler import (
    LogRecoCompositeHandler,
)
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)



class ArgannoPlusLogRecoProblem(LogRecoProblem, AnnotationProblem):
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
            # Assignment: Annotate a source text and logically reconstruct its main argument in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create a coherent argumentative text annotation and a corresponding logical argument reconstruction in standard form (premise-conclusion structure).

            ::: {{.source_text}}
            {sources}
            :::

                   
            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                   

            ## Argument Reconstruction Task Details                   

            Logically analyse and reconstruct the text's main argumentation as deductively valid inference with Argdown.

            - For each proposition in your reconstruction (premises and conclusions), provide an adequate FOL formalization in NLTK
              syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses.
              Only declare variables that are used in the corresponding formalization and that have not been declared before.
              Ensure that your formalizations are consistent with each other.

            - For each inference step in the argument, provide information about which previously introduced premises or 
              conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
            
            - Provide a succinct label (title) for each argument and summarize its gist in line with Argdown syntax conventions. 
                  
            Importantly, enclose your Argdown snippet in a fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

                   
            ## Required Coherence of Annotation and Argument Reconstruction                                                

            The argument reconstruction and the annotated source text must cohere with each other. There should be one-to-many correspondence between premises/conclusion(s) and annotated text segments. Moreover, the inferential relations in the reconstructed argument should reflect the annotated support relations.
                   
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
class ArgannoPlusLogReco(ArgannoPlusInfreco):
    """
    Solution to the ArgannoPlusLogReco problem: annotation and argdown snippet.
    
    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """
    @classmethod
    def from_raw_answer(cls, raw_answer):
        solution = super().from_raw_answer(raw_answer)
        return cls(
            annotated_source_text=solution.annotated_source_text,
            argdown_snippet=solution.argdown_snippet,
            unparsed_solution=solution.unparsed_solution,
        )


class ArgannoPlusLogRecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgannoPlusLogRecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + LogReco problem must be a string or a list of strings"
        )


class ArgannoPlusLogRecoJudge(MPJudge):
    """Judge for the anno plus argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgannoPlusLogRecoProblem), "Problem must be an ArgannoPlusLogRecoProblem"
        assert isinstance(original_solution, ArgannoPlusLogReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgannoPlusLogReco) for solution in solutions
        ), "All solutions must be ArgannoPlusLogReco objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgannoPlusLogRecoProblem), "Problem must be an ArgannoPlusLogRecoProblem"
        assert isinstance(solution, ArgannoPlusLogReco), "Solution must be an ArgannoPlusLogReco"

        infreco_handler = InfRecoCompositeHandler(
            handlers = [
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
                LogRecoCompositeHandler(),
                ArgannoInfrecoCoherenceHandler(),                
            ]
        )
        request = VerificationRequest(
            inputs=str(solution), source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation


