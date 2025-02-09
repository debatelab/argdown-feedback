import pprint
import random
from typing import Any
import pytest
import asyncio
import dataclasses

from argdown_hirp.base import (
    ChatMessage,
    Problem,
    Evaluation,
    Feedback,
    ChatPreferencePair,
    ProblemGenerator,
    SolutionGenerator,
    Judge,
    FeedbackGenerator,
    QualitativePreferencePairGenerator,
    HIRPreferencePairGenerator,
)


class NumberProblem(Problem):
    def __init__(self, number: int):
        self.number = number

    def cast(self, ask_for_invalid=False, hints: list[str] | None = None) -> str:
        prompt = f"Even: {self.number}? (y/n)"
        if ask_for_invalid:
            prompt += " (Invalid answer please!)"
        return prompt


@dataclasses.dataclass
class NumberSolution:
    answer: str

    def __str__(self):
        return self.answer


class NumberProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if not isinstance(inputs, int):
            raise ValueError("inputs must be an integer")
        return NumberProblem(inputs)


class NumberSolutionGenerator(SolutionGenerator):
    def __init__(self, mode: str, n_solutions: int):
        self.mode = mode
        self.n_solutions = n_solutions

    async def arun(
        self,
        problem: NumberProblem,
        original_solution: NumberSolution | None = None,
        feedback: Feedback | None = None,
    ) -> list[NumberSolution]:
        is_even = problem.number % 2 == 0
        ca = "y" if is_even else "n"
        fa = "n" if is_even else "y"
        cs = NumberSolution(ca)
        fs = NumberSolution(fa)
        if self.mode == "correct":
            return [cs] * self.n_solutions
        elif self.mode == "incorrect":
            return [fs] * self.n_solutions
        elif self.mode == "first_correct":
            return [cs] + [fs] * (self.n_solutions - 1)
        elif self.mode == "first_valid":
            return [NumberSolution("y")] + [NumberSolution("x")] * (
                self.n_solutions - 1
            )
        elif self.mode == "valid_after_magic":
            if feedback is not None and "magic" in feedback.feedback:
                return [NumberSolution("x")] * self.n_solutions
            else:
                return [cs] * self.n_solutions
        elif self.mode == "correct_after_magic":
            if feedback is not None and "magic" in feedback.feedback:
                return [fs] * self.n_solutions
            else:
                return [cs] * self.n_solutions

        return random.choices([cs, fs], k=self.n_solutions)


class NumberJudge(Judge):
    async def arun(
        self,
        problem: Problem,
        solutions: list[NumberSolution],
        original_solution: NumberSolution | None = None,
        feedback: Feedback | None = None,
    ) -> list[Evaluation]:
        assert isinstance(problem, NumberProblem), "problem must be a NumberProblem"
        is_even = problem.number % 2 == 0
        evaluations = []
        for solution in solutions:
            is_well_formed = solution.answer in ["y", "n"]
            if is_well_formed:
                is_correct = (solution.answer == "y") == is_even
            else:
                is_correct = False
            artifacts = {"is_correct": is_correct}
            evaluations.append(
                Evaluation(is_well_formed=is_well_formed, artifacts=artifacts)
            )
        return evaluations


class EmptyFeedbackGenerator(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Any,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        return [Feedback(feedback="empty", prompt="empty")]


class MagicFeedbackGenerator(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Any,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        return [Feedback(feedback="magic", prompt="magic")]


class NumberQualitativePreferencePairGenerator(QualitativePreferencePairGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        candidate_solutions: list[NumberSolution],
        evaluations: list[Evaluation],
        original_solution: NumberSolution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pairs = []
        top_correct_solution = next(
            (
                cs
                for cs, e in zip(candidate_solutions, evaluations)
                if e.artifacts["is_correct"]
            ),
            None,
        )
        top_incorrect_solution = next(
            (
                cs
                for cs, e in zip(candidate_solutions, evaluations)
                if not e.artifacts["is_correct"]
            ),
            None,
        )
        if top_correct_solution is not None and top_incorrect_solution is not None:
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
                        ChatMessage(
                            role="assistant", content=str(top_correct_solution)
                        ),
                    ],
                    rejected=[
                        ChatMessage(role="user", content=problem.cast()),
                        *intermediary_chat,
                        ChatMessage(
                            role="assistant", content=str(top_incorrect_solution)
                        ),
                    ],
                )
            )

        return pairs



def hirp_factory(mode: str, feedback_gen = EmptyFeedbackGenerator):
    return HIRPreferencePairGenerator(
        problem_generator=NumberProblemGenerator(),
        solution_generator=NumberSolutionGenerator(mode=mode, n_solutions=3),
        judge=NumberJudge(),
        feedback_generator=feedback_gen(),
        qualitative_preference_pair_generator=NumberQualitativePreferencePairGenerator(),
    )


@pytest.mark.asyncio
async def test_always_false():
    hirp_generator = hirp_factory(mode="incorrect")
    chat = await hirp_generator.arun(1)
    pprint.pprint(chat)
    assert not chat


@pytest.mark.asyncio
async def test_always_true():
    hirp_generator = hirp_factory(mode="correct")
    chat = await hirp_generator.arun(1)
    pprint.pprint(chat)
    assert not chat

@pytest.mark.asyncio
async def test_first_correct():
    hirp_generator = hirp_factory(mode="first_correct")
    chat = await hirp_generator.arun(1)
    pprint.pprint(chat)
    assert chat
    assert chat[0]["chosen"][-1]["content"] == "n"
    assert chat[0]["rejected"][-1]["content"] == "y"

@pytest.mark.asyncio
async def test_():
    hirp_generator = hirp_factory(mode="first_valid")
    chat = await hirp_generator.arun(1)
    pprint.pprint(chat)
    if chat:
        assert chat[0]["chosen"][-1]["content"] == "y"
        assert chat[0]["rejected"][-1]["content"] == "x"
        assert chat[1]["chosen"][-1]["content"] == "x"
        assert chat[1]["rejected"][-1]["content"] == "y"

