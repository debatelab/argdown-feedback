from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
)
from argdown_hirpo.tasks.compound.arganno_plus_infreco import (
    ArgannoPlusInfreco,
    ArgannoPlusInfrecoJudge,
)
from argdown_hirpo.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    AnnotationProblem,
)
from argdown_hirpo.tasks.core.logreco import (
    LogRecoProblem,
)
from argdown_hirpo.verifiers.logreco_verifier import LogRecoVerifier



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
                        
            Analyse the argumentation in the following **source text**. Create a a coherent argumentative text annotation and a corresponding logical argument reconstruction in standard form (premise-conclusion structure).

            ::: {{.source_text}}              
            {sources}
            :::

                   
            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way.
                        
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
        prompt = "Revise your previously submitted annotation and argument reconstruction given the above evaluation and feedback."

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


class ArgannoPlusLogRecoJudge(ArgannoPlusInfrecoJudge):
    """Judge for the anno plus argument mapping task."""

    def _evaluate_solution(self, problem, reco) -> Evaluation:
        evaluation = super()._evaluate_solution(problem, reco)
        argdown = evaluation.artifacts["argdown"]
        eval_data = evaluation.metrics
        is_valid = evaluation.is_valid
        if argdown is None:
            return evaluation
    
        eval_data["argument_unused_propositions"] = ""
        eval_data["argument_flawed_formalizations"] = ""
        eval_data["argument_invalid_inference"] = ""
        eval_data["argument_redundant_premises"] = ""
        eval_data["argument_inconsistent_premises"] = ""

        # unused propositions
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = LogRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.uses_all_props()
            if check is False:
                msgs.append(
                    f"Error in {argument.label}: {msg}" if msg else f"Unused propositions in argument {argument.label}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_unused_propositions"] = "\n".join(msgs)
        del msgs

        # check for syntactically correct formalizations
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = LogRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.has_flawless_formalizations()
            if check is False:
                msgs.append(
                    f"Error in {argument.label}: {msg}" if msg else f"Flawed formalizations in argument {argument.label}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_flawed_formalizations"] = "\n".join(msgs)

        # check for valid inference
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = LogRecoVerifier(argdown, argument_idx=argument_idx)
            for veri_fn in [
                verifier.is_globally_deductively_valid,
                verifier.is_locally_deductively_valid
            ]:
                check, msg = veri_fn()
                if check is False:
                    msgs.append(
                        f"Error in {argument.label}: {msg}"
                        if msg else f"Invalid inference in argument {argument.label}"
                    )
        if msgs:
            is_valid = False
            eval_data["argument_invalid_inference"] = "\n".join(msgs)
        del msgs

        # check for redundant premises
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = LogRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.all_premises_relevant()
            if check is False:
                msgs.append(
                    f"Error in {argument.label}: {msg}"
                    if msg else f"Redundant premises in argument {argument.label}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_redundant_premises"] = "\n".join(msgs)
        del msgs

        # check for inconsistent premises
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = LogRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.premises_consistent()
            if check is False:
                msgs.append(
                    f"Error in {argument.label}: {msg}"
                    if msg else f"Inconsistent premises in argument {argument.label}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_inconsistent_premises"] = "\n".join(msgs)
        del msgs

        evaluation.is_valid = is_valid
        evaluation.metrics = eval_data
        return evaluation

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgannoPlusLogRecoProblem), "Problem must be an ArgannoPlusLogRecoProblem"
        assert isinstance(original_solution, ArgannoPlusLogReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgannoPlusLogReco), (
                "All solutions must be ArgannoPlusLogReco"
            )
            evaluations.append(self._evaluate_solution(problem, solution))

        return evaluations

