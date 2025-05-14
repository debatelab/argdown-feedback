import pprint
from typing import Sequence
import pytest
import dataclasses

from argdown_feedback.tasks.base import (
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
    HIRPreferencePairGenerator,
)


class NumberProblem(Problem):
    def __init__(self, number: int):
        self.number = number

    def instruct_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation: Evaluation | None = None
    ) -> str:
        prompt = f"Even: {self.number}? (y/n)"
        if ask_for_invalid:
            prompt += " (Invalid answer please!)"
        if hints:
            prompt += " - ".join(hints)
        return prompt

    def revise_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation: Evaluation | None = None
    ) -> str:
        prompt = "Revise your answer given the feedback."
        if ask_for_invalid:
            prompt += " (Invalid answer please!)"
        if hints:
            prompt += " " + " - ".join(hints)
        return prompt


@dataclasses.dataclass
class NumberSolution(Solution):
    answer: str

    def __str__(self):
        return self.answer
    
    @classmethod
    def from_raw_answer(cls, answer):
        return answer


class NumberProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if not isinstance(inputs, int):
            raise ValueError("inputs must be an integer")
        return NumberProblem(inputs)


# Solution Generators
##############################

class NumberSolutionGenerator(SolutionGenerator):
    def __init__(self, n_solutions: int):
        self.n_solutions = n_solutions


class AllCorrectGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        is_even = problem.number % 2 == 0
        ca = "y" if is_even else "n"
        cs = NumberSolution(ca)
        return [cs] * self.n_solutions


class AllIncorrectGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        is_even = problem.number % 2 == 0
        fa = "n" if is_even else "y"
        fs = NumberSolution(fa)
        return [fs] * self.n_solutions
    

class FirstCorrectGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        is_even = problem.number % 2 == 0
        ca = "y" if is_even else "n"
        fa = "n" if is_even else "y"
        cs = NumberSolution(ca)
        fs = NumberSolution(fa)
        return [cs] + [fs] * (self.n_solutions - 1)
    

class YXGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        return [NumberSolution("y")] + [NumberSolution("x")] * (self.n_solutions - 1)


class YNXGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        return [
            NumberSolution(["y","n","x"][i % 3])
            for i in range(self.n_solutions)
        ]


class ValidAfterMagicGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        print(f"ValidAfterMagicGen: arun (problem={problem.instruct_prompt()}, original_solution={original_solution}, feedback={feedback})")
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        if feedback is not None and "magic" in feedback.feedback:
            return [NumberSolution("y")] + [NumberSolution(f"z{i}") for i in range(self.n_solutions-1)]
        else:
            return [NumberSolution(f"x{i}") for i in range(self.n_solutions)]


class CorrectAfterMagicGen(NumberSolutionGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[NumberSolution]:
        assert (
            isinstance(original_solution, NumberSolution) or original_solution is None
        )
        is_even = problem.number % 2 == 0
        ca = "y" if is_even else "n"
        fa = "n" if is_even else "y"
        cs = NumberSolution(ca)
        fs = NumberSolution(fa)
        if feedback is not None and "magic" in feedback.feedback:
            return [cs] * self.n_solutions
        else:
            return [fs] * self.n_solutions






class NumberJudge(Judge):
    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[Evaluation]:
        assert isinstance(problem, NumberProblem), "problem must be a NumberProblem"
        assert original_solution is None or isinstance(
            original_solution, NumberSolution
        ), "original_solution must be a NumberSolution"
        is_even = problem.number % 2 == 0
        evaluations = []
        for solution in solutions:
            assert isinstance(solution, NumberSolution), (
                "solution must be a NumberSolution"
            )
            is_valid = solution.answer in ["y", "n"]
            if is_valid:
                is_correct = (solution.answer == "y") == is_even
            else:
                is_correct = False
            artifacts = {"is_correct": is_correct}
            evaluations.append(
                Evaluation(is_valid=is_valid, artifacts=artifacts, metrics={})
            )
        return evaluations


class EmptyFeedbackGenerator(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        print("EmptyFeedbackGenerator: arun")
        return [Feedback(feedback="empty", prompt="empty")]


class MagicFeedbackGenerator(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        print(f"MagicFeedbackGenerator: arun ({solution})")
        return [Feedback(feedback="magic", prompt=f"magic: improve {solution}")]


class MixFeedbackGenerator(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        print(f"MagicFeedbackGenerator: arun ({solution})")
        return [
            Feedback(feedback="empty", prompt="empty"),
            Feedback(feedback="magic", prompt=f"magic: improve {solution}"),
        ]

class MixFeedbackGenerator2(FeedbackGenerator):
    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        print(f"MagicFeedbackGenerator: arun ({solution})")
        return [
            Feedback(feedback="magic", prompt=f"magic: improve {solution}"),
            Feedback(feedback="empty", prompt="empty"),
        ]


class NumberVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    async def arun(
        self,
        problem: NumberProblem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert all(isinstance(cs, NumberSolution) for cs in candidate_solutions)
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
            pairs.append(
                ChatPreferencePair(
                    chosen=ProblemSolutionChat(
                        problem=problem,
                        solution=top_correct_solution,
                        feedback=feedback,
                        original_solution=original_solution,
                    ).as_chat(hints=["mathematically perfect answer, please"]),
                    rejected=ProblemSolutionChat(
                        problem=problem,
                        solution=top_incorrect_solution,
                        feedback=feedback,
                        original_solution=original_solution,
                    ).as_chat(hints=["mathematically perfect answer, please"]),
                )
            )

        return pairs


def hirp_factory(solution_generator_class: type[NumberSolutionGenerator], feedback_gen=EmptyFeedbackGenerator):
    return HIRPreferencePairGenerator(
        problem_generator=NumberProblemGenerator(),
        solution_generator=solution_generator_class(n_solutions=3),
        judge=NumberJudge(),
        feedback_generator=feedback_gen(),
        virtue_preference_pair_generator=NumberVirtuePreferencePairGenerator(),
    )


@pytest.mark.asyncio
async def test_always_false():
    hirp_gen = hirp_factory(AllIncorrectGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    hirp_gen_magic_fbk = hirp_factory(AllIncorrectGen, MagicFeedbackGenerator)
    pairs, _ = await hirp_gen_magic_fbk.arun(1)
    pprint.pprint(pairs)
    assert not pairs


@pytest.mark.asyncio
async def test_always_true():
    hirp_generator = hirp_factory(AllCorrectGen)
    pairs, _ = await hirp_generator.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    hirp_gen_magic_fbk = hirp_factory(AllCorrectGen, MagicFeedbackGenerator)
    pairs, _ = await hirp_gen_magic_fbk.arun(1)
    pprint.pprint(pairs)
    assert not pairs


@pytest.mark.asyncio
async def test_first_correct():
    hirp_generator = hirp_factory(FirstCorrectGen)
    pairs, stats = await hirp_generator.arun(1)
    pprint.pprint(pairs)
    assert len(pairs) == 1  # since NumberVirtuePreferencePairGenerator returns one pair and doesn't do symmetric HIRP
    assert pairs[0]["chosen"][-1]["content"] == "n"
    assert pairs[0]["rejected"][-1]["content"] == "y"
    assert stats["n_total"] == 1


@pytest.mark.asyncio
async def test_first_valid():
    hirp_generator = hirp_factory(YXGen)
    pairs, _ = await hirp_generator.arun(1)
    pprint.pprint(pairs)
    if pairs:
        len(pairs) == 2  # symmetric HIRP
        assert pairs[0]["chosen"][-1]["content"] == "y"
        assert pairs[0]["rejected"][-1]["content"] == "x"
        assert pairs[1]["chosen"][-1]["content"] == "x"
        assert pairs[1]["rejected"][-1]["content"] == "y"

@pytest.mark.asyncio
async def test_correct_incorrect_invalid():
    hirp_generator = hirp_factory(YNXGen)

    # number = 1
    pairs, _ = await hirp_generator.arun(1)
    pprint.pprint(pairs)
    assert 1 <= len(pairs) <= 2 # depending on whether validity/virtue pairs are created
    if len(pairs) == 2:
        assert pairs[0]["chosen"][-1]["content"] == "y"
        assert pairs[0]["rejected"][-1]["content"] == "x"
        assert pairs[1]["chosen"][-1]["content"] == "x"
        assert pairs[1]["rejected"][-1]["content"] == "y"
    if len(pairs) == 1:
        assert pairs[0]["chosen"][-1]["content"] == "n"
        assert pairs[0]["rejected"][-1]["content"] == "y"

    # number = 2
    pairs, _ = await hirp_generator.arun(2)
    pprint.pprint(pairs)
    assert 1 <= len(pairs) <= 2 # depending on whether validity/virtue pairs are created
    if len(pairs) == 2:
        assert pairs[0]["chosen"][-1]["content"] == "y"
        assert pairs[0]["rejected"][-1]["content"] == "x"
        assert pairs[1]["chosen"][-1]["content"] == "x"
        assert pairs[1]["rejected"][-1]["content"] == "y"
    if len(pairs) == 1:
        assert pairs[0]["chosen"][-1]["content"] == "y"
        assert pairs[0]["rejected"][-1]["content"] == "n"

@pytest.mark.asyncio
async def test_valid_after_magic():
    hirp_gen = hirp_factory(ValidAfterMagicGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs
    
    # problem = 1
    hirp_gen_magic_fbk = hirp_factory(ValidAfterMagicGen, MagicFeedbackGenerator)
    pairs, _ = await hirp_gen_magic_fbk.arun(1)
    pprint.pprint(pairs)
    # as the feedback is always the same, we will never get feeback preferences
    assert not any("magic" in pairs[i]["chosen"][-1]["content"] for i in range(len(pairs)))
    assert all(len(pair["chosen"]) == 6 for pair in pairs)
    # moreover, as the revised answer are sometimes valid, but never correct (1 is not even)
    # we will get 2 or no pairs per original solution
    assert len(pairs) in [0, 2, 4, 6]
    # (in)valid answer is never preferred to (in)valid answer
    assert not any(
        pairs[i]["chosen"][-1]["content"] in ["y", "n"] == pairs[i]["rejected"][-1]["content"] in ["y", "n"]
        for i in range(len(pairs))
    )

@pytest.mark.asyncio
async def test_valid_after_magic2():
    hirp_gen = hirp_factory(ValidAfterMagicGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs
    
    # problem = 2
    hirp_gen_magic_fbk = hirp_factory(ValidAfterMagicGen, MagicFeedbackGenerator)
    pairs, _ = await hirp_gen_magic_fbk.arun(2)
    pprint.pprint(pairs)
    # as the feedback is always the same, we will never get feeback preferences
    assert not any("magic" in pairs[i]["chosen"][-1]["content"] for i in range(len(pairs)))
    assert all(len(pair["chosen"]) == 6 for pair in pairs)
    # moreover, as the revised answer are always correct if valid (2 is even)
    # we will get 2 or 1 chat pairs per original solution
    assert len(pairs) in [3, 4, 5, 6]
    # (in)valid answer is never preferred to (in)valid answer
    assert not any(
        pairs[i]["chosen"][-1]["content"] in ["y", "n"] == pairs[i]["rejected"][-1]["content"] in ["y", "n"]
        for i in range(len(pairs))
    )

@pytest.mark.asyncio
async def test_valid_after_magic3():
    hirp_gen = hirp_factory(ValidAfterMagicGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    # problem = 1
    hirp_gen_zero_magic_fbk = hirp_factory(ValidAfterMagicGen, MixFeedbackGenerator)
    pairs, _ = await hirp_gen_zero_magic_fbk.arun(1)
    pprint.pprint(pairs)
    assert pairs
    # as the first feedback is empty and the second is magic, we will get one feedback preference pair for each
    # original solution
    assert sum("magic" in pairs[i]["chosen"][-1]["content"] for i in range(len(pairs))) == 3
    assert sum(len(pair["chosen"]) == 4 for pair in pairs) == 3
    # but the "magic" feedback is never rejected
    assert not any("magic" in pairs[i]["rejected"][-1]["content"] for i in range(len(pairs)))
    # (in)valid answer is never preferred to (in)valid answer
    assert not any(
        pairs[i]["chosen"][-1]["content"] in ["y", "n"] == pairs[i]["rejected"][-1]["content"] in ["y", "n"]
        for i in range(len(pairs))
    )

@pytest.mark.asyncio
async def test_valid_after_magic4():
    hirp_gen = hirp_factory(ValidAfterMagicGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    # problem = 1
    hirp_gen_zero_magic_fbk = hirp_factory(ValidAfterMagicGen, MixFeedbackGenerator2)
    pairs, _ = await hirp_gen_zero_magic_fbk.arun(1)
    pprint.pprint(pairs)
    assert pairs
    # as the first feedback is magic and the second is empty, revision will stop after first feedback
    # we will get no feedback preference pairs for any original solution
    assert sum("magic" in pairs[i]["chosen"][-1]["content"] for i in range(len(pairs))) == 0
    assert sum(len(pair["chosen"]) == 4 for pair in pairs) == 0
    # but the "magic" feedback is never rejected
    assert not any("magic" in pairs[i]["rejected"][-1]["content"] for i in range(len(pairs)))
    # (in)valid answer is never preferred to (in)valid answer
    assert not any(
        pairs[i]["chosen"][-1]["content"] in ["y", "n"] == pairs[i]["rejected"][-1]["content"] in ["y", "n"]
        for i in range(len(pairs))
    )


@pytest.mark.asyncio
async def test_correct_after_magic():
    """
    if every initial solution is valid (whatever its virtue),
    we don't do self-criqitque.
    """

    hirp_gen = hirp_factory(CorrectAfterMagicGen)
    pairs, _ = await hirp_gen.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    hirp_gen_magic_fbk = hirp_factory(CorrectAfterMagicGen, MagicFeedbackGenerator)
    pairs, _ = await hirp_gen_magic_fbk.arun(1)
    pprint.pprint(pairs)
    assert not pairs

    hirp_gen_mix_fbk = hirp_factory(CorrectAfterMagicGen, MixFeedbackGenerator)

    pairs, _ = await hirp_gen_mix_fbk.arun(1)
    pprint.pprint(pairs)
    assert not pairs  # all solutions are valid, no pref pairs for training

    pairs, _ = await hirp_gen_mix_fbk.arun(2)
    pprint.pprint(pairs)
    assert not pairs  # all solutions are valid, no pref pairs for training

