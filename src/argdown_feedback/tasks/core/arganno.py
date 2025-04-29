import dataclasses
from textwrap import dedent
from typing import Sequence

from bs4 import BeautifulSoup

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    ScoringVirtuePreferencePairGenerator,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    FeedbackGenerator,
)

from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import HasAnnotationsHandler
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.base import CompositeHandler

ANNOTATION_SCHEME = dedent("""
    <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
    <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
    <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
    <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
    <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
    <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->
""")


class AnnotationProblem(Problem):
    """Task: Apply the argumentative annotation scheme to a text."""

    def __init__(self, sources: str | list[str], strip_html: bool = True):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        if strip_html:
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
            Assignment: Apply a given annotation scheme to a source text.
                        
            Annotate the following **source text** in order to identify the argumentative function of different parts in the text.

            ::: {{.source_text}}              
            {sources}
            :::

            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                        
            Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'.
        """)
            .strip()
            .format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

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
        prompt = (
            "Revise your previous annotation given the above evaluation and feedback."
        )

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
class Annotation(Solution):
    """Solution to the annotation problem: just an annotated text."""

    annotated_source_text: str

    def __str__(self):
        return self.annotated_source_text

    @classmethod
    def from_raw_answer(cls, answer) -> "Annotation":
        """Extract the annotated source text from the answer."""
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=answer)
        result = handler.process(request)
        code_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml and vr.code_snippet
            ),
            None,
        )
        code_snippet = code_snippet if code_snippet is not None else answer
        return cls(annotated_source_text=code_snippet)


class AnnotationProblemGenerator(ProblemGenerator):
    # TODO: Vary and configure the annotation problems generated
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return AnnotationProblem(inputs)
        raise ValueError(
            "Inputs to an annotation problem must be a string or a list of strings"
        )


class AnnotationJudge(MPJudge):
    """Judge for the annotation task."""

    def parse_xml_snippet(
        self, annotated_source_text: str
    ) -> tuple[BeautifulSoup, str | None]:
        error_msg: str | None = None
        ast = annotated_source_text.strip("\n ")
        if ast.startswith("```xml") and ast.endswith("```") and len(ast.splitlines()) > 1:
            ast = "\n".join(ast.splitlines()[1:-1])
        else:  # no fenced code block
            error_msg = "Failed to extract single fenced annotation block:"
            if ast.count("```xml") == 0:
                error_msg += " No fenced code block starting with '```xml'."
            if ast.count("```xml") > 1:
                error_msg += " More than one fenced code block starting with '```xml'."
            if not ast.endswith("```"):
                error_msg += " No closing '```'."

        multi_valued_attributes = {"*": {"supports", "attacks"}}
        soup = BeautifulSoup(
            ast,
            "html.parser",
            multi_valued_attributes=multi_valued_attributes,
        )
        return soup, error_msg

    def _check_inputs(self, problem, solutions, original_solution = None, feedback = None):
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(original_solution, Annotation) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        for solution in solutions:
            assert isinstance(solution, Annotation), "All solutions must be Annotations"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(solution, Annotation), "Solution must be an Annotation"

        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasAnnotationsHandler(),
                ArgannoCompositeHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=solution.annotated_source_text, source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation
    


# class AnnotationJudge2(Judge):
#     """Judge for the annotation task."""

#     def parse_xml_snippet(
#         self, annotated_source_text: str
#     ) -> tuple[BeautifulSoup, str | None]:
#         error_msg: str | None = None
#         ast = annotated_source_text.strip("\n ")
#         if ast.startswith("```xml") and ast.endswith("```") and len(ast.splitlines()) > 1:
#             ast = "\n".join(ast.splitlines()[1:-1])
#         else:  # no fenced code block
#             error_msg = "Failed to extract single fenced annotation block:"
#             if ast.count("```xml") == 0:
#                 error_msg += " No fenced code block starting with '```xml'."
#             if ast.count("```xml") > 1:
#                 error_msg += " More than one fenced code block starting with '```xml'."
#             if not ast.endswith("```"):
#                 error_msg += " No closing '```'."

#         multi_valued_attributes = {"*": {"supports", "attacks"}}
#         soup = BeautifulSoup(
#             ast,
#             "html.parser",
#             multi_valued_attributes=multi_valued_attributes,
#         )
#         return soup, error_msg

#     def _evaluate_annotation(
#         self, problem: AnnotationProblem, annotation: Annotation
#     ) -> Evaluation:
#         handler = CompositeHandler(
#             handlers=[
#                 DefaultProcessingHandler(),
#                 HasAnnotationsHandler(),
#                 ArgannoCompositeHandler(),
#             ]
#         )
#         request = VerificationRequest(
#             inputs=annotation.annotated_source_text, source=problem.sources
#         )
#         result = handler.process(request)
#         evaluation = Evaluation.from_verification_request(result)
#         return evaluation

#     async def arun(
#         self,
#         problem: Problem,
#         solutions: Sequence[Solution],
#         original_solution: Solution | None = None,
#         feedback: Feedback | None = None,
#     ) -> Sequence[Evaluation]:
#         assert isinstance(problem, AnnotationProblem), (
#             "Problem must be an AnnotationProblem"
#         )
#         assert isinstance(original_solution, Annotation) or original_solution is None
#         assert feedback or original_solution is None, (
#             "Feedback is required for evaluating revised solutions"
#         )

#         evaluations = []
#         for solution in solutions:
#             assert isinstance(solution, Annotation), "All solutions must be Annotations"
#             evaluations.append(self._evaluate_annotation(problem, solution))

#         return evaluations


class AnnotationFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_feedbacks", 5)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)

    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(solution, Annotation), "Solution must be an Annotation"
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given annotation.

            You will be shown an argumentative annotation problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

                                                
            ## Problem Statement
            {problem}

            
            ## Student's Solution
            {solution}

            
            ## Evaluation
            The student's solution is NOT valid.
            Particular issues:
            {evaluation_issues}

            
            Given this information, provide feedback to the student and clear instructions for how to improve the solution.
        """).format(
            problem=problem.instruct_prompt(),
            solution=str(solution),
            evaluation_issues=evaluation_issues,
        )

        answers = await self._generate(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
            n=self.n_feedbacks,
            temperature=self.temperature,
        )
        # remove empty and duplicate answers
        answers = [a for a in answers if a]
        answers = list(set(answers))

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class AnnotationScopePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of annotated proposition elements."""

    hints = ["Try to identify as many proposition elements as possible"]

    def _score(
        self, problem: Problem, annotation: Solution, evaluation: Evaluation
    ) -> float:
        return len(evaluation.artifacts["soup"].find_all("proposition"))


class AnnotationSupportsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of support relations between propositions."""

    hints = ["Try to identify as many support relations as possible"]

    def _score(
        self, problem: Problem, annotation: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        supports = sum(
            len(proposition.get("supports", [])) for proposition in propositions
        )
        return supports


class AnnotationAttacksPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of attack relations between propositions."""

    hints = ["Try to identify as many attack / disconfirmation relations as possible"]

    def _score(
        self, problem: Problem, annotation: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", [])) for proposition in propositions
        )
        return attacks


class AnnotationNoAttacksPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with smallest number of attack relations between propositions."""

    hints = ["Avoid using attack / disconfirmation relations"]

    def _score(
        self, problem: Problem, annotation: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", [])) for proposition in propositions
        )
        return 1 / (1 + attacks)


class AnnotationCoveragePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger coverage of source text."""

    hints = ["Try to cover as much of the source text as possible"]

    def _score(
        self, problem: Problem, annotation: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        coverage = sum(len(proposition.get_text()) for proposition in propositions)
        return coverage
