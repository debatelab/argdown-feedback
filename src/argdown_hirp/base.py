"Base HIR preference pair generators."

from abc import ABC, abstractmethod
import dataclasses
import random
from statistics import mean
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatPreferencePair(TypedDict):
    chosen: list[ChatMessage]
    rejected: list[ChatMessage]


class Problem(ABC):
    """A problem."""

    @abstractmethod
    def cast(self, ask_for_invalid=False, hints: list[str] | None = None) -> str:
        """Cast the problem as a problem statement, including an instruction how to solve it."""


@dataclasses.dataclass
class Evaluation:
    """An evaluation of a solution."""

    is_well_formed: bool
    artifacts: dict[str, Any]


@dataclasses.dataclass
class Feedback:
    prompt: str
    feedback: str


class HIRAbstractGenerator(ABC):
    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        pass


class ProblemGenerator(HIRAbstractGenerator):
    """Generates a problem."""

    @abstractmethod
    async def arun(self, inputs) -> Problem:
        pass


class SolutionGenerator(HIRAbstractGenerator):
    """Generates solutions."""

    @abstractmethod
    async def arun(
        self, Problem, original_solution: Any = None, feedback: Feedback | None = None
    ) -> list[Any]:
        pass


class Judge(HIRAbstractGenerator):
    """Judges solutions."""

    @abstractmethod
    async def arun(
        self,
        problem: Problem,
        solutions: list[Any],
        original_solution: Any = None,
        feedback: Feedback | None = None,
    ) -> list[Evaluation]:
        pass


class FeedbackGenerator(HIRAbstractGenerator):
    """Generates feedback."""

    @abstractmethod
    async def arun(self, problem: Problem, solution: Any, evaluation: Evaluation) -> list[Feedback]:
        pass


class QualitativePreferencePairGenerator(HIRAbstractGenerator):
    """Generates preference pairs from qualitative properties."""

    @abstractmethod
    async def arun(
        self,
        problem,
        candidate_solutions,
        evaluations,
        original_solution: Any = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pass


class HIRPreferencePairGenerator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        qualitative_preference_pair_generator: QualitativePreferencePairGenerator,
        feedback_generator: FeedbackGenerator | None = None,
        n_solutions: int = 10,
        n_feedbacks: int = 5,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        self.feedback_generator = feedback_generator
        self.qualitative_preference_pair_generator = (
            qualitative_preference_pair_generator
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def syntc_prefpair(
        self,
        problem: Problem,
        candidate_solutions: list[Any],
        evaluations: list[Evaluation],
        original_solution: Any = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pairs: list[ChatPreferencePair] = []

        top_valid_solution = next(
            cs for cs, e in zip(candidate_solutions, evaluations) if e.is_well_formed
        )
        top_invalid_solution = next(
            cs
            for cs, e in zip(candidate_solutions, evaluations)
            if not e.is_well_formed
        )

        intermediary_chat = []
        if original_solution is not None and feedback is not None:
            intermediary_chat += [
                ChatMessage(
                    role="assistant",
                    content=str(original_solution),
                ),
                ChatMessage(
                    role="user",
                    content=feedback.prompt,
                ),
            ]

        pairs.append(
            ChatPreferencePair(
                chosen=[
                    ChatMessage(role="user", content=problem.cast()),
                    *intermediary_chat,
                    ChatMessage(role="assistant", content=str(top_valid_solution)),
                ],
                rejected=[
                    ChatMessage(role="user", content=problem.cast()),
                    *intermediary_chat,
                    ChatMessage(role="assistant", content=str(top_invalid_solution)),
                ],
            )
        )

        pairs.append(
            ChatPreferencePair(
                chosen=[
                    ChatMessage(
                        role="user", content=problem.cast(ask_for_invalid=True)
                    ),
                    *intermediary_chat,
                    ChatMessage(role="assistant", content=str(top_invalid_solution)),
                ],
                rejected=[
                    ChatMessage(
                        role="user", content=problem.cast(ask_for_invalid=True)
                    ),
                    *intermediary_chat,
                    ChatMessage(role="assistant", content=str(top_valid_solution)),
                ],
            )
        )

        return pairs

    async def build_pref_pair(
        self,
        problem: Problem,
        candidate_solutions: list[Any],
        evaluations: list[Evaluation],
        original_solution: Any = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Candidate solutions and evaluations must have the same length."
        )
        syntactic_validity = mean(int(e.is_well_formed) for e in evaluations)
        if random.random() < syntactic_validity:
            print("Constructing qualitative preference pairs")
            return await self.qualitative_preference_pair_generator.arun(
                problem,
                candidate_solutions,
                evaluations,
                original_solution=original_solution,
                feedback=feedback,
            )
        else:
            print("Constructing syntactic preference pairs")
            return self.syntc_prefpair(
                problem,
                candidate_solutions,
                evaluations,
                original_solution=original_solution,
                feedback=feedback,
            )

    async def self_critique(
        self,
        problem: Problem,
        candidate_solutions: list[Any],
        evaluations: list[Evaluation],
    ) -> list[ChatPreferencePair]:
        pairs: list[ChatPreferencePair] = []
        if self.feedback_generator is None:
            return pairs

        for cs, e in zip(candidate_solutions, evaluations):
            if not e.is_well_formed:
                feedbacks = await self.feedback_generator.arun(problem, cs, e)
                revisions: list[list[Any]] = []
                revision_evaluations: list[list[Evaluation]] = []
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

                    if not any(e.is_well_formed for e in revision_evals):
                        continue

                    pairs.extend(
                        await self.build_pref_pair(
                            problem,
                            candidate_revisions,
                            revision_evals,
                            original_solution=cs,
                            feedback=feedback,
                        )
                    )
                # get most and least successful feedback by counting valid revisions
                n_valid_revisions = [
                    sum(e.is_well_formed for e in re) for re in revision_evaluations
                ]
                if n_valid_revisions and max(n_valid_revisions) > min(
                    n_valid_revisions
                ):
                    most_successful_feedback = feedbacks[
                        n_valid_revisions.index(max(n_valid_revisions))
                    ]
                    least_successful_feedback = feedbacks[
                        n_valid_revisions.index(min(n_valid_revisions))
                    ]
                    pairs.append(
                        ChatPreferencePair(
                            chosen=[
                                ChatMessage(role="user", content=problem.cast()),
                                ChatMessage(role="assistant", content=cs),
                                ChatMessage(
                                    role="user", content=most_successful_feedback.prompt
                                ),
                                ChatMessage(
                                    role="assistant",
                                    content=most_successful_feedback.feedback,
                                ),
                            ],
                            rejected=[
                                ChatMessage(role="user", content=problem.cast()),
                                ChatMessage(role="assistant", content=cs),
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

    async def arun(self, inputs) -> list[ChatPreferencePair]:
        problem = await self.problem_generator.arun(inputs)
        print(problem)
        candidate_solutions = await self.solution_generator.arun(problem)
        print(candidate_solutions)
        evaluations = await self.judge.arun(problem, candidate_solutions)
        print(evaluations)

        if all(not e.is_well_formed for e in evaluations):
            return await self.self_critique(problem, candidate_solutions, evaluations)

        return await self.build_pref_pair(problem, candidate_solutions, evaluations)
