"Base HIR preference pair generators."

from abc import ABC, abstractmethod
import dataclasses
import logging
import random
from statistics import mean
from typing import Any, Sequence, TypedDict

from openai import AsyncOpenAI, OpenAI
import tenacity

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
        self, ask_for_invalid=False, hints: list[str] | None = None
    ) -> str:
        """Cast the problem as a problem statement, including an instruction to solve it."""

    @abstractmethod
    def revise_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None
    ) -> str:
        """Instruction to revise earlier mentioned solution of the problem."""


class Solution(ABC):
    """Abstract base class representing a solution."""

    @abstractmethod
    def __str__(self) -> str:
        """Cast the solution as a string."""


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
    artifacts: dict[str, Any]


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

    def as_chat(self, **problem_kwargs) -> list[ChatMessage]:
        if self.original_solution is None or self.feedback is None:
            return [
                ChatMessage(
                    role="user", content=self.problem.instruct_prompt(**problem_kwargs)
                ),
                ChatMessage(role="assistant", content=str(self.solution)),
            ]
        return [
            ChatMessage(role="user", content=self.problem.instruct_prompt()),
            ChatMessage(role="assistant", content=str(self.original_solution)),
            ChatMessage(role="user", content=self.feedback.prompt),
            ChatMessage(role="assistant", content=self.feedback.feedback),
            ChatMessage(
                role="user", content=self.problem.revise_prompt(**problem_kwargs)
            ),
            ChatMessage(role="assistant", content=str(self.solution)),
        ]


# GENERATORS
######################


class HIRAbstractGenerator(ABC):
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
                self.model_id = next(models.data[0].id)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=60),
        stop=tenacity.stop_after_attempt(6),
    )
    async def _generate(self, messages, **gen_kwargs):
        stream = False
        completion = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=stream,
            **gen_kwargs,
        )
        answers = [choice.message.content for choice in completion.choices]
        return answers


class ProblemGenerator(HIRAbstractGenerator):
    """Generates a problem."""

    @abstractmethod
    async def arun(self, inputs) -> Problem:
        pass


class SolutionGenerator(HIRAbstractGeneratorLLM):
    """Generates solutions."""

    @abstractmethod
    async def arun(
        self,
        Problem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Solution]:
        pass


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


class FeedbackGenerator(HIRAbstractGeneratorLLM):
    """Generates feedback."""

    @abstractmethod
    async def arun(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> Sequence[Feedback]:
        pass


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


# MAIN GENERATOR
######################


class HIRPreferencePairGenerator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        virtue_preference_pair_generator: VirtuePreferencePairGenerator,
        feedback_generator: FeedbackGenerator | None = None,
        failure_type_preference_pair_generator: FailureTypePreferencePairGenerator
        | None = None,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        self.feedback_generator = feedback_generator
        self.virtue_preference_pair_generator = virtue_preference_pair_generator
        self.failure_type_preference_pair_generator = (
            failure_type_preference_pair_generator
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validity_vs_virtue_router(
        self, mean_syntactic_validity: float
    ) -> tuple[bool, bool]:
        do_virtue_hirp = random.random() < mean_syntactic_validity
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
            cs for cs, e in zip(candidate_solutions, evaluations) if e.is_valid
        )
        top_invalid_solution = next(
            cs for cs, e in zip(candidate_solutions, evaluations) if not e.is_valid
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(),
            )
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True),
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
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Candidate solutions and evaluations must have the same length."
        )
        pairs: list[ChatPreferencePair] = []
        mean_syntactic_validity = mean(int(e.is_valid) for e in evaluations)
        do_validity_hirp, do_virtue_hirp = self.validity_vs_virtue_router(
            mean_syntactic_validity
        )
        if do_virtue_hirp:
            logger.debug("Constructing virtue preference pair")
            pairs.extend(
                await self.virtue_preference_pair_generator.arun(
                    problem,
                    candidate_solutions,
                    evaluations,
                    original_solution=original_solution,
                    feedback=feedback,
                )
            )
        if do_validity_hirp:
            logger.debug("Constructing syntactic validity preference pair")
            pairs.extend(
                self.build_validity_pref_pair(
                    problem,
                    candidate_solutions,
                    evaluations,
                    original_solution=original_solution,
                    feedback=feedback,
                )
            )
        return pairs

    async def run_self_critique(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
    ) -> list[ChatPreferencePair]:
        pairs: list[ChatPreferencePair] = []
        if self.feedback_generator is None:
            return pairs

        for cs, e in zip(candidate_solutions, evaluations):
            if e.is_valid:
                continue
            feedbacks = await self.feedback_generator.arun(problem, cs, e)
            revisions: list[Sequence[Solution]] = []
            revision_evaluations: list[Sequence[Evaluation]] = []
            for feedback in feedbacks:
                candidate_revisions = await self.solution_generator.arun(
                    problem, original_solution=cs, feedback=feedback
                )
                revision_evals = await self.judge.arun(
                    problem,
                    candidate_revisions,
                    original_solution=cs,
                    feedback=feedback,
                )
                revisions.append(candidate_revisions)
                revision_evaluations.append(revision_evals)

                # run revision-specific HIRP to generate solution preference pairs

                if not any(re.is_valid for re in revision_evals):
                    if self.failure_type_preference_pair_generator is not None:
                        pairs.extend(
                            await self.failure_type_preference_pair_generator.arun(
                                problem, candidate_solutions, evaluations
                            )
                        )
                    continue

                pairs.extend(
                    await self.build_solution_pref_pair(
                        problem,
                        candidate_revisions,
                        revision_evals,
                        original_solution=cs,
                        feedback=feedback,
                    )
                )

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

        return pairs

    def self_critique_router(self, evaluations: Sequence[Evaluation]) -> tuple[bool, bool]:
        """router for main workflow"""
        do_self_critique = not any(e.is_valid for e in evaluations)
        return do_self_critique, not do_self_critique

    async def arun(self, inputs) -> list[ChatPreferencePair]:
        """main workflow logic"""
        problem = await self.problem_generator.arun(inputs)
        candidate_solutions = await self.solution_generator.arun(problem)
        evaluations = await self.judge.arun(problem, candidate_solutions)

        pairs: list[ChatPreferencePair] = []

        do_self_critique, skip_self_critique = self.self_critique_router(evaluations)

        if do_self_critique:
            pairs = await self.run_self_critique(
                problem, candidate_solutions, evaluations
            )

        if skip_self_critique:
            pairs = await self.build_solution_pref_pair(
                problem, candidate_solutions, evaluations
            )

        if not pairs and self.failure_type_preference_pair_generator is not None:
            pairs = await self.failure_type_preference_pair_generator.arun(
                problem, candidate_solutions, evaluations
            )

        return pairs
