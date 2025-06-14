"Base HIR preference pair generators."

from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ProcessPoolExecutor
import copy
import dataclasses
import functools
import hashlib
import logging
import random
from statistics import mean
from textwrap import dedent
from typing import Any, Awaitable, Callable, Sequence, TypedDict

from bs4 import BeautifulSoup
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pyargdown import Argdown
import tenacity

from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)

logger = logging.getLogger(__name__)


# DATA MODELS
######################


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatPreferencePair(TypedDict):
    chosen: list[ChatMessage]
    rejected: list[ChatMessage]


class Problem(ABC):
    """Abstract base class representing a problem."""

    @abstractmethod
    def instruct_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation=None
    ) -> str:
        """Cast the problem as a problem statement, including an instruction to solve it."""

    @abstractmethod
    def revise_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation=None
    ) -> str:
        """Instruction to revise earlier mentioned solution of the problem."""

    def ask_for_invalid_prompt(self, prompt: str, evaluation) -> str:
        """string transformation that reverts prompt instruction, asking llm to make specific mistakes"""
        warning_text = (
            "> [!WARNING]\n"
            "> For didactic purposes, I want you to make mistakes in your answer.\n"
        )
        prompt = warning_text + "\n\n" + prompt + "\n\n" + warning_text

        if evaluation:
            metrics = {k: v for k, v in evaluation.metrics.items() if v}
            if metrics:
                prompt += "> Expected errors:\n"
                for k, msg in metrics.items():
                    # format multi-line arror messages
                    msg = "\n".join([f">   {line}" for line in str(msg).splitlines()])
                    msg = msg[4:]  # rm ">   " in first line
                    prompt += f"> - {k}: {msg}\n"

        return prompt

    def ask_for_invalid_revise_prompt(self, prompt: str, evaluation) -> str:
        """string transformation that reverts revise prompt instruction, asking llm to make specific mistakes"""
        prompt = prompt + (
            "\n\n"
            "> [!WARNING]\n"
            "> For didactic purposes, I still want you to make mistakes in your revised answer, violating the above instructions.\n"
        )

        if evaluation:
            metrics = {k: v for k, v in evaluation.metrics.items() if v}
            if metrics:
                prompt += "> Expected errors:\n"
                for k, msg in metrics.items():
                    # format multi-line arror messages
                    msg = "\n".join([f">   {line}" for line in str(msg).splitlines()])
                    msg = msg[4:]  # rm ">   " in first line
                    prompt += f"> - {k}: {msg}\n"

        return prompt


class Solution(ABC):
    """Abstract base class representing a solution."""

    @abstractmethod
    def __str__(self) -> str:
        """Cast the solution as a string."""

    @abstractmethod
    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""

    @classmethod
    @abstractmethod
    def from_raw_answer(cls, raw_answer: str) -> "Solution":
        """Cast a raw answer as a solution."""


@dataclasses.dataclass
class Evaluation:
    """
    Evaluation of a solution

    Every solution is valid or invalid. -- Which can mean different things
    depending on the problem, and the criteria used to evaluate it.

    The ability to generate *valid* solutions -- in which ever way validity
    is interpreted -- is the primary skill HIRPO seeks to instill in LLMs.

    On top of marking a solution as valid or invalid, the evaluation may
    contain additional information about the solution, or evaluation artifacts.
    Such additional information, artifacts or metrics may be used by the
    feedback generator or by the virtue preference pair generator.
    """

    is_valid: bool
    artifacts: dict[str, Any]  # global artifacts
    metrics: dict[str, Any]

    @classmethod
    def from_verification_request(
        cls,
        request: VerificationRequest,
    ) -> "Evaluation":
        """
        Create an Evaluation from a VerificationRequest.
        """

        metrics = dict(
            (f"{e + 1:02d}_{result.verifier_id}", result.message)
            for e, result in enumerate(request.results)
        )

        artifacts = {}

        last_soup = next(
            (
                data
                for data in reversed(request.verification_data)
                if data.dtype == VerificationDType.xml
                and data.data is not None
                and isinstance(data.data, BeautifulSoup)
            ),
            None,
        )
        artifacts["soup"] = last_soup.data if last_soup is not None else None

        verification_data_argdown = [
            data
            for data in request.verification_data
            if data.dtype == VerificationDType.argdown
            and data.data is not None
            and isinstance(data.data, Argdown)
        ]

        last_argdown = next(reversed(verification_data_argdown), None)
        artifacts["argdown"] = last_argdown.data if last_argdown is not None else None

        # argdown_map
        last_argdown_map = next(
            (
                data
                for data in reversed(verification_data_argdown)
                if data.metadata and data.metadata.get("filename") == "map.ad"
            ),
            None,
        )
        artifacts["argdown_map"] = (
            last_argdown_map.data if last_argdown_map is not None else None
        )

        # argdown_reco
        last_argdown_reco = next(
            (
                data
                for data in reversed(verification_data_argdown)
                if data.metadata
                and data.metadata.get("filename") == "reconstructions.ad"
            ),
            None,
        )
        artifacts["argdown_reco"] = (
            last_argdown_reco.data if last_argdown_reco is not None else None
        )

        # formalizations are stored as details in result of WellFormedFormulasHandler
        all_expressions = None
        all_declarations = None
        if last_argdown is not None:
            argdown_vd_id = (
                last_argdown_reco.id
                if last_argdown_reco is not None
                else last_argdown.id
            )
            wff_result = next(
                (
                    result
                    for result in request.results
                    if "WellFormedFormulasHandler"
                    in result.verifier_id  # NOTE: hacky way to get right VerificationResult
                    and result.verification_data_references == [argdown_vd_id]
                ),
                None,
            )
            if wff_result is not None:
                all_expressions = wff_result.details.get("all_expressions")
                all_declarations = wff_result.details.get("all_declarations")
        artifacts["all_expressions"] = all_expressions
        artifacts["all_declarations"] = all_declarations

        return cls(
            is_valid=request.is_valid(),
            artifacts=artifacts,
            metrics=metrics,
        )


@dataclasses.dataclass
class Feedback:
    prompt: str
    feedback: str


@dataclasses.dataclass
class ProblemSolutionChat:
    """
    Dataclass representing a problem-solving chat.
    Starts with the problem presentation, and concludes with the final solution.
    Possibly contains a revision phase (original solution and feedback) inbetween.
    If original_solution and feedback are not None, solution represents the final,
    revised solution.
    """

    problem: Problem
    solution: Solution
    original_solution: Solution | None = None
    feedback: Feedback | None = None

    def as_chat(self, use_raw_answer: bool = False, **problem_kwargs) -> list[ChatMessage]:
        answer_content = str(self.solution) if not use_raw_answer else self.solution.raw_answer()
        if self.original_solution is None or self.feedback is None:
            return [
                ChatMessage(
                    role="user", content=self.problem.instruct_prompt(**problem_kwargs)
                ),
                ChatMessage(role="assistant", content=answer_content),
            ]
        return [
            ChatMessage(role="user", content=self.problem.instruct_prompt()),
            ChatMessage(role="assistant", content=str(self.original_solution)),
            ChatMessage(role="user", content=self.feedback.prompt),
            ChatMessage(role="assistant", content=self.feedback.feedback),
            ChatMessage(
                role="user", content=self.problem.revise_prompt(**problem_kwargs)
            ),
            ChatMessage(role="assistant", content=answer_content),
        ]


# GENERATORS
######################


class HIRAbstractGenerator(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        pass


class HIRAbstractGeneratorLLM(ABC):
    def __init__(self, *args, **kwargs):
        inference_base_url = kwargs.get("inference_base_url", None)
        model_id = kwargs.get("model_id", None)
        if inference_base_url and model_id:
            self.model_id = model_id
            self.client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=inference_base_url,
            )
            models = OpenAI(api_key="EMPTY", base_url=inference_base_url).models.list()
            if self.model_id not in [d.id for d in models.data]:
                logging.getLogger(__name__).warning(
                    f"Model {self.model_id} not found in OpenAI API. Model served: {models}. Will proceed with first of these models."
                )
                self.model_id = next(iter(models.data)).id

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=30),
        stop=tenacity.stop_after_attempt(6),
    )
    async def _generate(self, messages, **gen_kwargs):
        stream = False
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stream=stream,
                **gen_kwargs,
            )
            answers = [
                choice.message.content
                for choice in completion.choices
                if choice.message.content is not None
            ]
            return answers
        except Exception as e:
            if isinstance(e, BadRequestError):
                if "maximum context length" in e.message:
                    hash = hashlib.sha256(str(messages).encode()).hexdigest()
                    logger.warning(
                        f"Request with hash {hash} is exceeding maximum context length. Will not retry."
                    )
                    logger.debug(f"Error message: {str(e)}")
                    return []
            logger.error(f"Error calling the inference server: {str(e)}")
            logger.debug("Error-inducing messages:")
            for message in messages:
                logger.debug(f"  {message['role']}: {message['content']}")
            logger.debug(f"Error-inducing kwargs: {gen_kwargs}")
            raise e


class ProblemGenerator(HIRAbstractGenerator):
    """Generates a problem."""

    @abstractmethod
    async def arun(self, inputs) -> Problem:
        pass


class ProblemGeneratorLLM(HIRAbstractGeneratorLLM, ProblemGenerator):
    """Generates a problem."""


class SolutionGenerator(HIRAbstractGeneratorLLM):
    """Generates solutions."""

    @abstractmethod
    async def arun(
        self,
        problem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Solution]:
        pass


class GenericSolutionGenerator(SolutionGenerator):
    """Generic solution generator with postprocessing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # solution class
        self.solution_class = kwargs.get("solution_class", Solution)
        assert self.solution_class, "Solution class is required"
        assert issubclass(self.solution_class, Solution), (
            "Solution class must be a subclass of Solution"
        )
        # generation kwargs
        self.n_solutions = kwargs.get("n_solutions", 10)
        n_revisions = max(round(self.n_solutions / 2), 1)
        self.n_revisions = kwargs.get("n_revisions", n_revisions)
        self.remove_duplicates = kwargs.get("remove_duplicates", True)
        self.gen_kwargs = kwargs.get("gen_kwargs", {"max_tokens": 4096})

    def remove_repetitions(
        self, answer: str, keep: int = 3, min_lines: int = 16
    ) -> str:
        """
        Remove repetitive blocs of text at the end of the answer.
        """
        lines = answer.split("\n")
        if len(lines) <= min_lines:
            return answer

        # remove one-line-blocs
        # print(set(lines[(-2-keep):-1]))
        while (
            len(set(lines[(-2 - keep) : -1])) == 1
            and lines[-2].startswith(lines[-1])
            and len(lines) > min_lines
            and lines[(-2)].strip()
        ):
            del lines[(-2)]  # delete line at idx -2

        # remove multi-line-blocs
        for bloclength in [2, 3]:
            while (
                len(
                    set(
                        [
                            "".join(lines[-ri - bloclength : -ri])
                            for ri in range(1, bloclength * (keep + 1), bloclength)
                        ]
                    )
                )
                == 1
                and lines[-1 - bloclength].startswith(lines[-1])
                and len(lines) > min_lines
                and any(lines[-ri].strip() for ri in range(2, bloclength + 1))
            ):
                del lines[-1 - bloclength : -1]  # delete last bloc

        return "\n".join(lines)  # return the modified answer

    async def arun(
        self,
        problem: Problem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Solution]:
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

        n = self.n_solutions if original_solution is None else self.n_revisions

        answers = await self._generate(
            messages,
            n=n,
            **self.gen_kwargs,
        )

        # remove empty and duplicate answers
        answers = [a for a in answers if a and isinstance(a, str)]
        answers = list(set(answers)) if self.remove_duplicates else answers

        # remove repetitive blocs of text at the end of the answer
        answers = [self.remove_repetitions(answer) for answer in answers]

        recos: list[Solution] = []

        for answer in answers:
            reco = self.solution_class.from_raw_answer(answer)
            recos.append(reco)  # type: ignore

        return recos


class Judge(HIRAbstractGenerator):
    """Judges solutions."""

    @abstractmethod
    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        pass


class MPJudge(Judge):
    """
    MPJudge implements parallel multiprocessing of solutions to improve efficiency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = kwargs.get("max_workers", 8)

    @abstractmethod
    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        """
        Check that the inputs of verification request are correct and consistent.
        """

    @staticmethod
    @abstractmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        """Evaluate a given solution."""
        pass

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        self._check_inputs(
            problem,
            solutions,
            original_solution=original_solution,
            feedback=feedback,
        )

        # multiprocessing with concurrent.futures.PoolExecutor

        # prepare partial function
        evaluate_solution = functools.partial(
            self._evaluate_solution,
            problem=problem,
            original_solution=original_solution,
            feedback=feedback,
        )

        # evaluate solutions in parallel
        loop = asyncio.get_event_loop()

        tasks = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for solution in solutions:
                tasks.append(
                    loop.run_in_executor(executor, evaluate_solution, solution)
                )

        # wait for all tasks to finish
        evaluations = await asyncio.gather(*tasks)

        return evaluations


class FeedbackGenerator(HIRAbstractGeneratorLLM):
    """Generates feedback."""

    @abstractmethod
    async def arun(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> Sequence[Feedback]:
        pass


class GenericFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_feedbacks", 5)
        self.gen_kwargs = kwargs.get(
            "gen_kwargs", {"temperature": 0.8, "max_tokens": 1024}
        )

    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given argument map.

            You will be shown an argument analysis problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

            ########################                                    
            ## Problem Statement 
            {problem}

            ########################
            ## Student's Solution
            {solution}

            ########################
            ## Evaluation 
            The student's solution is NOT valid.
            Particular issues:
            {evaluation_issues}

            
            Given this information, provide feedback to the student and succinct instructions for how to improve the solution.
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
            n=self.n_feedbacks,
            **self.gen_kwargs,
        )
        # remove empty and duplicate answers
        answers = [a for a in answers if a]
        answers = list(set(answers))

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class VirtuePreferencePairGenerator(HIRAbstractGenerator):
    """
    Generates preference pairs from differences in terms of additional
    virtues other than (syntactic) validity of candidate_solutions.
    """

    @abstractmethod
    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pass


class ScoringVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the informal argument reconstruction task
    based on score."""

    hints: list[str] = []

    @abstractmethod
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        pass

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )

        pairs: list[ChatPreferencePair] = []

        valid_recos: list[tuple[Solution, Evaluation]] = [
            (solution, evaluation)
            for solution, evaluation in zip(candidate_solutions, evaluations)
            if evaluation.is_valid
        ]
        if len(valid_recos) < 2:
            return pairs

        # rank valid recos according to the _score function
        valid_recos.sort(key=lambda x: self._score(problem, x[0], x[1]), reverse=True)

        top_score = self._score(problem, *valid_recos[0])
        if top_score == self._score(problem, *valid_recos[-1]):
            return pairs

        top_reco, _ = valid_recos[0]
        weaker_reco = random.choice(
            [s for s, e in valid_recos if self._score(problem, s, e) < top_score]
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=weaker_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
            )
        )

        return pairs


class FailureTypePreferencePairGenerator(HIRAbstractGenerator):
    """
    Generates preference pairs from failure type differences
    between candidate solutions.
    """

    @abstractmethod
    async def arun(
        self,
        problem,
        candidate_solutions,
        evaluations,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pass


class GenericFailureDiffPreferencePairGenerator(FailureTypePreferencePairGenerator):
    """Generate failure-type-preference pairs based on the differences in failure profiles."""

    avoid_errors_hints = [
        (
            "Very important! It is acceptable if you make these kinds of errors: {common_errors}. "
            "But no matter whether your solution is flawless or not, "
            "you should definitely avoid the following mistakes:\n"
        ),
        (
            "> [!WARNING]\n"
            "> For didactic purposes, I want you to make mistakes in your answer.\n"
            "> Specifically, I expect your answer to contain the following mistakes: {common_errors}.\n"
            "> However, I want you to AVOID the following errors:\n"
        )
    ]

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )
        pairs: list[ChatPreferencePair] = []

        # consolidate metric keys
        metrics_batch: list[dict[str, Any]] = []
        for evaluation in evaluations:
            metrics: dict[str, Any] = {}
            for k, v in evaluation.metrics.items():
                key = k.split("_", 1)[1] if "_" in k else k
                if metrics.get(key) is None:
                    metrics[key] = v
                elif v:
                    metrics[key] = metrics[key] + " " + v
            metrics_batch.append(metrics)

        # count error types
        error_counts: dict[str, int] = {}
        for key in set([k for m in metrics_batch for k in m.keys()]):
            error_counts[key] = len([1 for metrics in metrics_batch if metrics.get(key)])
        common_errors: list[str] = [
            k for k, v in error_counts.items() if v == len(evaluations)
        ]
        # dismiss errors that are always or never avoided, i.e. always present
        error_counts = {
            k: v for k, v in error_counts.items() if 0 < v < len(evaluations)
        }

        if not error_counts:
            return pairs

        # get error type that is most common
        most_frequent_type, _ = max(
            error_counts.items(),
            key=lambda x: x[1],
        )

        # chosen solution avoids most common error
        chosen_idx = random.choice(
            [
                idx
                for idx, metrics in enumerate(metrics_batch)
                if not metrics.get(most_frequent_type)
            ]
        )
        # rejected solution commits most common error
        rejected_idx = random.choice(
            [
                idx
                for idx, metrics in enumerate(metrics_batch)
                if metrics.get(most_frequent_type)
            ]
        )

        # list all errors avoided by the chosen solution
        errors_avoided = {
            k: v
            for k, v in metrics_batch[rejected_idx].items()
            if v and not metrics_batch[chosen_idx].get(k)
        }

        avoid_errors_hint = random.choice(self.avoid_errors_hints)
        avoid_errors_hint = avoid_errors_hint.format(
            common_errors=", ".join(common_errors) if common_errors else "none"
        )
        hint = avoid_errors_hint + "\n".join(
            f"- {k}: {v}" for k, v in errors_avoided.items()
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=candidate_solutions[chosen_idx],
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=[hint]),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=candidate_solutions[rejected_idx],
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=[hint]),
            )
        )

        return pairs


# MAIN GENERATORS
######################


@dataclasses.dataclass
class HIRPOGenStats:
    """
    Statistics for preference pairs generated with HIRPOGen.
    """

    n_solutions: int = 0
    n_solutions_valid: int = 0
    n_total: int = 0
    n_validity_preference_pairs: int = 0
    n_virtue_preference_pairs: int = 0
    n_validity_preference_pairs_rev: int = 0
    n_virtue_preference_pairs_rev: int = 0
    n_feedback_preference_pairs: int = 0
    n_failure_type_preference_pairs: int = 0
    n_original_revision_pairs: int = 0

    # define operator overloading for addition
    def __add__(self, other: "HIRPOGenStats") -> "HIRPOGenStats":
        if not isinstance(other, HIRPOGenStats):
            raise TypeError("Can only add HIRPOGenStats objects")
        return HIRPOGenStats(
            n_solutions=self.n_solutions + other.n_solutions,
            n_solutions_valid=self.n_solutions_valid + other.n_solutions_valid,
            n_total=self.n_total + other.n_total,
            n_validity_preference_pairs=self.n_validity_preference_pairs
            + other.n_validity_preference_pairs,
            n_virtue_preference_pairs=self.n_virtue_preference_pairs
            + other.n_virtue_preference_pairs,
            n_validity_preference_pairs_rev=self.n_validity_preference_pairs_rev
            + other.n_validity_preference_pairs_rev,
            n_virtue_preference_pairs_rev=self.n_virtue_preference_pairs_rev
            + other.n_virtue_preference_pairs_rev,
            n_feedback_preference_pairs=self.n_feedback_preference_pairs
            + other.n_feedback_preference_pairs,
            n_failure_type_preference_pairs=self.n_failure_type_preference_pairs
            + other.n_failure_type_preference_pairs,
            n_original_revision_pairs=self.n_original_revision_pairs
            + other.n_original_revision_pairs,
        )


class HIRPreferencePairGenerator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        virtue_preference_pair_generator: VirtuePreferencePairGenerator
        | list[VirtuePreferencePairGenerator]
        | None = None,
        feedback_generator: FeedbackGenerator | None = None,
        failure_type_preference_pair_generator: FailureTypePreferencePairGenerator
        | None = None,
        ask_for_invalid_probs: dict[str, float] | None = None,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        self.feedback_generator = feedback_generator
        if virtue_preference_pair_generator is None:
            virtue_preference_pair_generator = []
        if isinstance(virtue_preference_pair_generator, VirtuePreferencePairGenerator):
            virtue_preference_pair_generator = [virtue_preference_pair_generator]
        self.virtue_preference_pair_generators = virtue_preference_pair_generator
        self.failure_type_preference_pair_generator = (
            failure_type_preference_pair_generator
        )
        self.ask_for_invalid_probs: dict[str, float] = ask_for_invalid_probs if ask_for_invalid_probs is not None else {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validity_vs_virtue_router(
        self, mean_syntactic_validity: float
    ) -> tuple[bool, bool]:
        do_virtue_hirp = (
            bool(self.virtue_preference_pair_generators)
            and random.random() < mean_syntactic_validity**4
        )
        do_validity_hirp = not do_virtue_hirp
        return do_validity_hirp, do_virtue_hirp

    def build_validity_pref_pair(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        """
        Builds two preference pairs based on syntactic differences
        between candidate_solutions, varies the instruction in
        accordance with HIR.
        """
        pairs: list[ChatPreferencePair] = []

        top_valid_solution = next(
            (cs for cs, e in zip(candidate_solutions, evaluations) if e.is_valid), None
        )
        top_invalid_solution, top_invalid_evaluation = next(
            (
                (cs, e)
                for cs, e in zip(candidate_solutions, evaluations)
                if not e.is_valid
            ),
            (None, None),
        )

        if top_valid_solution is None or top_invalid_solution is None:
            return pairs

        # current accuracy
        mean_syntactic_validity = mean(int(e.is_valid) for e in evaluations)

        # ask_for_invalid_probability is minimum of
        # - (1 - accuracy) and
        # - any configured ask_for_invalid_probability configured for any error_types 
        #   (e.g. HasAnnotationsHandler) present in top_invalid_solution
        # In consequence, if the current accuracy is high, the probability that the model
        # is shown an ask-for-invalid preference pair is low.
        min_ask_for_invalid_probability = min(
            [
                # select configured error-specific ask_for_invalid probability
                afi_prob
                for error_type, afi_prob in self.ask_for_invalid_probs.items()
                # if error_type has been made in top_invalid_solution
                if any(
                    error_type in key for key in top_invalid_evaluation.metrics.keys()  # type: ignore
                )
            ]
            + [1.0 - mean_syntactic_validity]
        )
        # randomly determine whether to do ask_for_invalid_pair
        add_ask_for_invalid_pair = random.random() < min_ask_for_invalid_probability

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(use_raw_answer=True), # we're including full raw answers here
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(use_raw_answer=True), # we're including full raw answers here
            )
        )

        if not add_ask_for_invalid_pair:
            return pairs

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True, evaluation=top_invalid_evaluation),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True, evaluation=top_invalid_evaluation),
            )
        )

        return pairs

    async def build_solution_pref_pair(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> tuple[list[ChatPreferencePair], HIRPOGenStats]:
        assert len(candidate_solutions) == len(evaluations), (
            "Candidate solutions and evaluations must have the same length."
        )
        pairs: list[ChatPreferencePair] = []
        stats = HIRPOGenStats()
        mean_syntactic_validity = mean(int(e.is_valid) for e in evaluations)
        do_validity_hirp, do_virtue_hirp = self.validity_vs_virtue_router(
            mean_syntactic_validity
        )
        if do_virtue_hirp:
            logger.debug("Constructing virtue preference pair")
            assert self.virtue_preference_pair_generators, (
                "Internal error: Attempting do_virtue_hirp while no virtue preference pair generators available."
            )
            shuffled_generators = copy.deepcopy(self.virtue_preference_pair_generators)
            random.shuffle(shuffled_generators)
            virtue_pairs: list[ChatPreferencePair] = []
            for virtue_preference_pair_generator in shuffled_generators:
                virtue_pairs = await virtue_preference_pair_generator.arun(
                    problem,
                    candidate_solutions,
                    evaluations,
                    original_solution=original_solution,
                    feedback=feedback,
                )
                if virtue_pairs:
                    break
            pairs.extend(virtue_pairs)
            stats.n_total += len(virtue_pairs)
            if original_solution is None:
                stats.n_virtue_preference_pairs += len(virtue_pairs)
            else:
                stats.n_virtue_preference_pairs_rev += len(virtue_pairs)

        if do_validity_hirp:
            logger.debug("Constructing syntactic validity preference pair")
            new_pairs = self.build_validity_pref_pair(
                problem,
                candidate_solutions,
                evaluations,
                original_solution=original_solution,
                feedback=feedback,
            )
            pairs.extend(new_pairs)
            stats.n_total += len(new_pairs)
            if original_solution is None:
                stats.n_validity_preference_pairs += len(new_pairs)
            else:
                stats.n_validity_preference_pairs_rev += len(new_pairs)

        return pairs, stats

    async def run_self_critique(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        generate_revision_failure_type_pairs: bool = False,
    ) -> tuple[list[ChatPreferencePair], HIRPOGenStats]:
        """self-critique branch"""

        async def run_selfcritique_workflow(
            problem: Problem, cs: Solution, e: Evaluation
        ) -> tuple[list[ChatPreferencePair], HIRPOGenStats]:
            """runs a full selfcritique workflow for a _single_ candidate_solution and its evaluation"""

            async def run_revision_workflow(
                problem: Problem, cs: Solution, e: Evaluation, feedback: Feedback
            ) -> tuple[list[ChatPreferencePair], Sequence[Evaluation], HIRPOGenStats]:
                """generates and evaluates revisions, construct pref pairs for single solution, evaluation, and feedback"""
                pairs_rev_wf: list[ChatPreferencePair] = []
                stats = HIRPOGenStats()
                candidate_revisions = await self.solution_generator.arun(
                    problem, original_solution=cs, feedback=feedback
                )
                if not candidate_revisions:
                    logger.info(
                        "No candidate revisions generated. Skipping revision workflow."
                    )
                    return pairs_rev_wf, [], stats
                try:
                    revision_evals = await self.judge.arun(
                        problem,
                        candidate_revisions,
                        original_solution=cs,
                        feedback=feedback,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to evaluate revisions ({str(exc)}). Skipping revision workflow."
                    )
                    return pairs_rev_wf, [], stats

                # run revision-specific HIRP to generate solution preference pairs

                if not any(re.is_valid for re in revision_evals):
                    if (
                        generate_revision_failure_type_pairs
                        and self.failure_type_preference_pair_generator is not None
                    ):
                        new_pairs = (
                            await self.failure_type_preference_pair_generator.arun(
                                problem, candidate_revisions, revision_evals
                            )
                        )
                        pairs_rev_wf.extend(new_pairs)
                        stats.n_total += len(new_pairs)
                        stats.n_failure_type_preference_pairs += len(new_pairs)
                    return pairs_rev_wf, revision_evals, stats

                if all(re.is_valid for re in revision_evals):
                    new_pairs, _ = await self.build_solution_pref_pair(
                        problem,
                        candidate_solutions=[cs, candidate_revisions[0]],
                        evaluations=[e, revision_evals[0]],
                    )
                    pairs_rev_wf.extend(new_pairs)
                    stats.n_total += len(new_pairs)
                    stats.n_original_revision_pairs += len(new_pairs)

                new_pairs, new_stats = await self.build_solution_pref_pair(
                    problem,
                    candidate_revisions,
                    revision_evals,
                    original_solution=cs,
                    feedback=feedback,
                )
                pairs_rev_wf.extend(new_pairs)
                stats += new_stats

                return pairs_rev_wf, revision_evals, stats

            pairs: list[ChatPreferencePair] = []
            stats = HIRPOGenStats()
            revision_evaluations: list[Sequence[Evaluation]] = []

            # generate feedbacks
            if self.feedback_generator is None:
                return pairs, stats
            feedbacks = await self.feedback_generator.arun(problem, cs, e)
            # revisions: list[Sequence[Solution]] = []

            # coros = [
            #     run_revision_workflow(problem, cs, e, feedback)
            #     for feedback in feedbacks
            # ]
            # for pairs_rev_wf, revision_evals in await asyncio.gather(*coros):
            #     revision_evaluations.append(revision_evals)
            #     pairs.extend(pairs_rev_wf)

            for feedback in feedbacks:
                (
                    pairs_rev_wf,
                    revision_evals,
                    stats_rev_wf,
                ) = await run_revision_workflow(problem, cs, e, feedback)
                revision_evaluations.append(revision_evals)
                pairs.extend(pairs_rev_wf)
                stats += stats_rev_wf

                # stop if we have at least one valid revision
                if any(ev.is_valid for ev in revision_evals):
                    break

            # discard all feedbacks that have not been used
            feedbacks = feedbacks[: len(revision_evaluations)]


            # generate feedback preference pair

            # get most and least successful feedback by counting valid revisions
            n_valid_revisions = [
                sum(e.is_valid for e in re) for re in revision_evaluations
            ]
            if n_valid_revisions and max(n_valid_revisions) > min(n_valid_revisions):
                most_successful_feedback = feedbacks[
                    n_valid_revisions.index(max(n_valid_revisions))
                ]
                least_successful_feedback = feedbacks[
                    n_valid_revisions.index(min(n_valid_revisions))
                ]
                pairs.append(
                    ChatPreferencePair(
                        chosen=[
                            ChatMessage(role="user", content=problem.instruct_prompt()),
                            ChatMessage(role="assistant", content=str(cs)),
                            ChatMessage(
                                role="user", content=most_successful_feedback.prompt
                            ),
                            ChatMessage(
                                role="assistant",
                                content=most_successful_feedback.feedback,
                            ),
                        ],
                        rejected=[
                            ChatMessage(role="user", content=problem.instruct_prompt()),
                            ChatMessage(role="assistant", content=str(cs)),
                            ChatMessage(
                                role="user", content=most_successful_feedback.prompt
                            ),
                            ChatMessage(
                                role="assistant",
                                content=least_successful_feedback.feedback,
                            ),
                        ],
                    )
                )
                stats.n_total += 1
                stats.n_feedback_preference_pairs += 1

            return pairs, stats

        pairs_all: list[ChatPreferencePair] = []
        stats_all = HIRPOGenStats()
        if self.feedback_generator is None:
            return [], stats_all

        # coros = [
        #     run_selfcritique_workflow(problem, cs, e)
        #     for cs, e in zip(candidate_solutions, evaluations)
        #     if not e.is_valid
        # ]
        # for p in await asyncio.gather(*coros):
        #     pairs_all.extend(p)

        for cs, e in zip(candidate_solutions, evaluations):
            if not e.is_valid:
                pairs, stats = await run_selfcritique_workflow(problem, cs, e)
                pairs_all.extend(pairs)
                stats_all += stats

        # Deduplicate (ignoring differences in rejected conversations)
        random.shuffle(pairs_all)
        pairs_all = [
            pair
            for i, pair in enumerate(pairs_all)
            if not any(p["chosen"] == pair["chosen"] for p in pairs_all[:i])
        ]
        # NOTE
        # We're only adjusting total counts
        # In consequence, pair_type_counts include duplicates and might not add up to total count
        stats_all.n_total = len(pairs_all)

        return pairs_all, stats_all

    def self_critique_router(
        self, evaluations: Sequence[Evaluation]
    ) -> tuple[bool, bool]:
        """router for main workflow"""
        do_self_critique = not any(e.is_valid for e in evaluations)
        return do_self_critique, not do_self_critique

    async def arun(self, inputs) -> tuple[list[ChatPreferencePair], dict[str, int]]:
        """main workflow logic"""
        problem = await self.problem_generator.arun(inputs)
        candidate_solutions = await self.solution_generator.arun(problem)
        evaluations = await self.judge.arun(problem, candidate_solutions)

        pairs: list[ChatPreferencePair] = []
        stats = HIRPOGenStats(
            n_solutions=len(evaluations),
            n_solutions_valid=len([e for e in evaluations if e.is_valid])
        )

        do_self_critique, skip_self_critique = self.self_critique_router(evaluations)

        if do_self_critique:
            try:
                pairs, stats = await self.run_self_critique(
                    problem, candidate_solutions, evaluations
                )
            except Exception as e:
                logger.error(f"Error in self-critique workflow: {e}")
                logger.info("Attempting to build pref pairs without self-critique.")
                skip_self_critique = True

        if skip_self_critique:
            pairs, stats = await self.build_solution_pref_pair(
                problem, candidate_solutions, evaluations
            )

        if not pairs and self.failure_type_preference_pair_generator is not None:
            pairs = await self.failure_type_preference_pair_generator.arun(
                problem, candidate_solutions, evaluations
            )
            stats = HIRPOGenStats(
                n_failure_type_preference_pairs=len(pairs),
                n_total=len(pairs),
            )

        return pairs, dataclasses.asdict(stats)


class HIREvaluator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def arun(
        self, inputs, log_samples_callback: Callable[..., Awaitable[None]] | None = None
    ) -> dict[str, Any]:
        """evaluate inputs and returns average accuracy of the candidate solutions"""
        problem = await self.problem_generator.arun(inputs)
        candidate_solutions = await self.solution_generator.arun(problem)
        evaluations = await self.judge.arun(problem, candidate_solutions)
        if log_samples_callback is not None:
            try:
                await log_samples_callback(
                    problem=problem.instruct_prompt(),
                    solutions=[str(s) for s in candidate_solutions],
                    evaluations=[
                        [
                            ("is_valid", str(e.is_valid)),
                            *[
                                (k, str(v))
                                for k, v in e.metrics.items()
                                if v is not None
                            ],
                        ]
                        for e in evaluations
                    ],
                )
            except Exception as e:
                logger.error(
                    f"Failed to log generated eval samples in log_samples_callback: {e}"
                )
        metric_keys = set(
            key for evaluation in evaluations for key in evaluation.metrics.keys()
        )
        eval_result: dict[str, Any] = {}
        for key in metric_keys:
            counts = sum(bool(e.metrics.get(key)) for e in evaluations)
            eval_result[f"{key}_counts"] = counts
        eval_result["n_solutions"] = len(evaluations)
        eval_result["accuracy"] = (
            sum(int(e.is_valid) for e in evaluations) / len(evaluations)
            if evaluations
            else 0.0
        )
        return eval_result
