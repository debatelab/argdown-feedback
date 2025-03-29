from abc import abstractmethod
import dataclasses
from difflib import unified_diff
from textwrap import dedent, shorten
from typing import Sequence

from bs4 import BeautifulSoup

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ChatPreferencePair,
    ProblemSolutionChat,
    ProblemGenerator,
    SolutionGenerator,
    Judge,
    FeedbackGenerator,
    VirtuePreferencePairGenerator,
)

ANNOTATION_SCHEME = dedent("""
    <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
    <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
    <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
    <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
    <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- label of argument or thesis in external argdown document -->
    <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- item label of premise or conclusion in external argdown argument -->
""")


class AnnotationProblem(Problem):
    """Task: Apply the argumentative annotation scheme to a text."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        sources = BeautifulSoup(sources, "html.parser").get_text()
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources

    def instruct_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation: Evaluation | None = None
    ) -> str:
        prompt = dedent("""
            Assignment: Apply a given annotation scheme to a source text.
                        
            Annotate the following **source text** in order to identify the argumentative function of different parts in the text.

            ::: {{.source_text}}              
            {sources}
            :::

            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way.
                        
            Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'.
        """).strip().format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)

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
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation: Evaluation | None = None
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


class AnnotationSolutionGenerator(SolutionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_solutions = kwargs.get("n_solutions", 10)
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_tokens = kwargs.get("max_tokens", 2048)

    async def arun(
        self,
        problem: AnnotationProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Annotation]:
        assert isinstance(original_solution, Annotation) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for revised solutions"
        )

        messages = [
            {
                "role": "user",
                "content": problem.instruct_prompt(),
            }
        ]

        if original_solution and feedback:
            messages += [
                {
                    "role": "assistant",
                    "content": str(original_solution),
                },
                {
                    "role": "user",
                    "content": feedback.prompt,
                },
                {
                    "role": "assistant",
                    "content": feedback.feedback,
                },
                {
                    "role": "user",
                    "content": problem.revise_prompt(),
                },
            ]

        answers = await self._generate(
            messages,
            max_tokens=self.max_tokens,
            n=self.n_solutions,
            temperature=self.temperature,
        )

        annotations = []

        # postprocess: extract fenced code block
        for answer in answers:
            if answer.count("```xml") == 1:
                if answer.split("```xml")[1].count("\n```") == 1:
                    answer = answer.split("```xml")[1].split("\n```")[0]
                    answer = "```xml" + answer + "\n```"
            annotations.append(Annotation(annotated_source_text=answer))

        return annotations

class AnnotationJudge(Judge):
    """Judge for the annotation task."""

    def parse_xml_snippet(self, annotated_source_text: str) -> tuple[BeautifulSoup, str | None]:

        error_msg: str | None = None
        ast = annotated_source_text.strip("\n ")
        if ast.startswith("```xml") and ast.endswith("```"):
            ast = "\n".join(ast.splitlines()[1:-1])
        else: # no fenced code block
            error_msg = "Failed to extract single fenced annotation block:"
            if ast.count("```xml") == 0:
                error_msg += " No fenced code block starting with '```xml'."
            if ast.count("```xml") > 1:
                error_msg += " More than one fenced code block starting with '```xml'."
            if "```\n" not in ast:
                error_msg += " No closing '```'."

        multi_valued_attributes = {"*": {"supports", "attacks"}}
        soup = BeautifulSoup(
            ast,
            "html.parser",
            multi_valued_attributes=multi_valued_attributes,
        )
        return soup, error_msg

    def _evaluate_annotation(
        self, problem: AnnotationProblem, annotation: Annotation
    ) -> Evaluation:
        is_valid = True
        eval_data = {
            "fenced_code_block": "",
            "nested_propositions": "",
            "missing_id": "",
            "duplicate_id": "",
            "invalid_support_ids": "",
            "invalid_attack_ids": "",
            "unknown_attributes": "",
        }

        # parse xml
        soup, error_msg = self.parse_xml_snippet(annotation.annotated_source_text)
        if error_msg:
            is_valid = False
            eval_data["fenced_code_block"] = error_msg
        del error_msg

        # Source text must not be altered (except for annotations and white space)
        lines_o = " ".join(problem.sources.split()).splitlines(keepends=True)
        lines_a = " ".join(soup.get_text().split()).splitlines(keepends=True)
        lines_o = [line for line in lines_o if line.strip()]
        lines_a = [line for line in lines_a if line.strip()]

        diff = list(unified_diff(lines_o, lines_a, n=0))
        if diff:
            is_valid = False
            eval_data["altered_source_text"] = (
                "Source text was altered. Diff:\n" + "".join(diff)
            )

        # No nested proposition annotations
        for proposition in soup.find_all("proposition"):
            if proposition.find_all("proposition"):  # type: ignore
                is_valid = False
                eval_data["nested_propositions"] = (
                    f"Nested annotations in proposition '{shorten(str(proposition), 256)}'"
                )
                break

        # Every proposition must have an id
        for proposition in soup.find_all("proposition"):
            if not proposition.get("id"):  # type: ignore
                is_valid = False
                eval_data["missing_id"] = (
                    f"Missing id in proposition '{shorten(str(proposition), 64)}'"
                )
                break

        # Every proposition must have a unique id
        ids = [
            proposition.get("id")  # type: ignore
            for proposition in soup.find_all("proposition")
        ]  # type: ignore
        duplicates = {id for id in ids if ids.count(id) > 1}
        if duplicates:
            is_valid = False
            eval_data["duplicate_id"] = f"Duplicate ids: {duplicates}"

        # Every "supports" reference must be a valid id
        for proposition in soup.find_all("proposition"):
            for support in proposition.get("supports", []):  # type: ignore
                if support not in ids:
                    is_valid = False
                    eval_data["invalid_support_ids"] = (
                        f"Supported proposition with id '{support}' in proposition '{shorten(str(proposition), 64)}' does not exist"
                    )
                    break

        # Every "attacks" reference must be a valid id
        for proposition in soup.find_all("proposition"):
            for attack in proposition.get("attacks", []):  # type: ignore
                if attack not in ids:
                    is_valid = False
                    eval_data["invalid_attack_ids"] = (
                        f"Attacked proposition with id '{attack}' in proposition '{shorten(str(proposition), 64)}' does not exist"
                    )
                    break

        # No unknown attributes
        for proposition in soup.find_all("proposition"):
            for attr in proposition.attrs:  # type: ignore
                if attr not in {
                    "id",
                    "supports",
                    "attacks",
                    "argument_label",
                    "ref_reco_label",
                }:
                    is_valid = False
                    eval_data["unknown_attributes"] = (
                        f"Unknown attribute '{attr}' in proposition '{shorten(str(proposition), 64)}'"
                    )
                    break

        # No unknown elements
        for element in soup.find_all():
            if element.name not in {"proposition"}:  # type: ignore
                is_valid = False
                eval_data["unknown_elements"] = (
                    f"Unknown element '{element.name}' at '{shorten(str(element), 64)}'"  # type: ignore
                )

        return Evaluation(
            is_valid=is_valid,
            artifacts={
                "soup": soup,
            },
            metrics=eval_data,
        )

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert (
            isinstance(original_solution, Annotation) or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, Annotation), (
                "All solutions must be Annotations"
            )
            evaluations.append(self._evaluate_annotation(problem, solution))

        return evaluations


class AnnotationFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_solutions", 5)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

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
            f"- **{k}**: {v}"
            for k, v in evaluation.metrics.items()
            if v
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
        """).format(problem=problem.instruct_prompt(), solution=str(solution), evaluation_issues=evaluation_issues)

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

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class AnnotationVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task."""

    hints: list[str] = []

    @abstractmethod
    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        pass

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert all(isinstance(s, Annotation) for s in candidate_solutions), (
            "All solutions must be Annotations"
        )
        assert original_solution is None or isinstance(original_solution, Annotation), (
            "Original solution must be an Annotation"
        )
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )

        pairs: list[ChatPreferencePair] = []

        # rank valid annotations according to the _score function
        valid_annotations = list(zip(candidate_solutions, evaluations))
        valid_annotations.sort(
            key=lambda x: self._score(x[0], x[1]), reverse=True
        )
        valid_annotations = [
            (solution, evaluation)
            for solution, evaluation in valid_annotations
            if evaluation.is_valid
        ]

        if len(valid_annotations) < 2:
            return pairs
        if self._score(*valid_annotations[0]) == self._score(*valid_annotations[-1]):
            return pairs

        top_annotation, _ = valid_annotations[0]
        bottom_annotation, _ = valid_annotations[-1]

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_annotation,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=bottom_annotation,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
            )
        )

        return pairs


class AnnotationScopePreferencePairGenerator(AnnotationVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of annotated proposition elements."""

    hints = ["Try to identify as many proposition elements as possible"]

    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        return len(evaluation.artifacts["soup"].find_all("proposition"))


class AnnotationSupportsPreferencePairGenerator(AnnotationVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of support relations between propositions."""

    hints = ["Try to identify as many support relations as possible"]

    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        supports = sum(
            len(proposition.get("supports", []))
            for proposition in propositions
        )
        return supports


class AnnotationAttacksPreferencePairGenerator(AnnotationVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of attack relations between propositions."""

    hints = ["Try to identify as many attack / disconfirmation relations as possible"]

    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", []))
            for proposition in propositions
        )
        return attacks


class AnnotationNoAttacksPreferencePairGenerator(AnnotationVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with smallest number of attack relations between propositions."""

    hints = ["Avoid using attack / disconfirmation relations"]

    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", []))
            for proposition in propositions
        )
        return 1/(1+attacks)

class AnnotationCoveragePreferencePairGenerator(AnnotationVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger coverage of source text."""

    hints = ["Try to cover as much of the source text as possible"]

    def _score(self, annotation: Solution, evaluation: Evaluation) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        coverage = sum(
            len(proposition.get_text())
            for proposition in propositions
        )
        return coverage
