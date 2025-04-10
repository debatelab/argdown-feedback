import pytest
import textwrap

from argdown_feedback.tasks.base import Feedback, Solution, GenericSolutionGenerator
from argdown_feedback.tasks.core.infreco import (
    InfRecoJudge,
    InfRecoProblem,
    InformalReco,
)
from argdown_feedback.tasks.core.logreco import (
    LogicalReco,
    LogRecoJudge,
    LogRecoFeedbackGenerator,
    FormalizationsFaithfulnessPreferencePairGenerator,
)
from argdown_feedback.tasks.sequential.logreco_from_infreco import (
    LogrecoFromInfrecoProblem,
    LogrecoFromInfrecoProblemGenerator,
    InfrecoProximityPreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return LogrecoFromInfrecoProblem

@pytest.fixture
def problem_generator_class():
    return LogrecoFromInfrecoProblemGenerator

@pytest.fixture
def solution_class():
    return LogicalReco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return LogRecoJudge

@pytest.fixture
def feedback_generator_class():
    return LogRecoFeedbackGenerator


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)
    ]


@pytest.fixture
def example_problem() -> LogrecoFromInfrecoProblem:
        argdown_snippet = textwrap.dedent("""
            ```argdown
            <Argument 1>: Animals suffer.

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        infreco_evaluation = InfRecoJudge()._evaluate_infreco(InfRecoProblem(argdown_snippet), InformalReco(argdown_snippet))
        return LogrecoFromInfrecoProblem(
            argdown_snippet=argdown_snippet,
            argdown_infreco=infreco_evaluation.artifacts.get("argdown"),
            infreco_evaluation=infreco_evaluation,
        ) 



@pytest.fixture
def valid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument 1>: Animals suffer.

            (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
            (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "eating it is wrong"}}
            -- {from: ["1", "2"]} --
            (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```
            <Argument 1>: Animals suffer.

            (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
            (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "eating it is wrong"}}
            -- {from: ["1", "2"]} --
            (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            Missing declaration in (2)

            ```argdown
            <Argument 1>: Animals suffer.

            (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
            (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))"}
            -- {from: ["1", "2"]} --
            (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
            ```
            """)
        ),

    ]


@pytest.fixture
def feedback1() -> Feedback:
    return Feedback(
        prompt="Please provide feedback.",
        feedback=textwrap.dedent("""
        **Feedback:**
        1. The solution provided does not follow the required formatting conventions. Specifically, \
        the argdown code block is not opened with '```argdown'.
                                 
        **Instructions for Improvement:**
        1. Start the codeblock with '```argdown'.
        """),
    )


def test_avail(model_kwargs):
    llm_available()


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_problem_generator(
    problem_generator_class,
    problem_class,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_problem_generator(
        problem_generator_class,
        problem_class,
        source_texts,
        model_kwargs,
        keeps_source_texts=False,
    )



@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_solution_generator(
    problem_generator_class,
    solution_generator_class,
    solution_class,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_solution_generator(
        problem_generator_class,
        solution_generator_class,
        solution_class,
        source_texts,
        model_kwargs
    )


@pytest.mark.asyncio
async def test_judge_valid(
    example_problem,
    judge_class,
    valid_recos,
):
    await HirpoTester.test_judge_valid2(
        example_problem,
        judge_class,
        valid_recos,
    )


@pytest.mark.asyncio
async def test_judge_invalid(
    example_problem,
    judge_class,
    invalid_recos,
):
    await HirpoTester.test_judge_invalid2(
        example_problem,
        judge_class,
        invalid_recos,
    )


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_feedback_generator(
    problem_generator_class,
    judge_class,
    feedback_generator_class,
    invalid_recos,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_feedback_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        invalid_recos,
        source_texts,
        model_kwargs
    )


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_revised_solution_generator(
    problem_generator_class,
    judge_class,
    feedback_generator_class,
    solution_generator_class,
    solution_class,
    invalid_recos,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_revised_solution_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        solution_generator_class,
        solution_class,
        invalid_recos,
        source_texts,
        model_kwargs
    )


@pytest.mark.asyncio
class TestInfRecoPreferencePairGenerators:

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
            (
                InfrecoProximityPreferencePairGenerator,
                """
                ```argdown
                <Suffering>: Animals suffer.

                (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
                (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "eating it is wrong"}}
                -- {from: ["1", "2"]} --
                (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
                ```
                """,
                """
                ```argdown
                <Leiden>: Animals suffer.

                (1) Tiere leiden. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
                (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "eating it is wrong"}}
                -- {from: ["1", "2"]} --
                (3) Tiere zu essen ist falsch. {formalization: "all x.(F(x) -> H(x))"}
                ```
                """,
            ),
            (
                FormalizationsFaithfulnessPreferencePairGenerator,
                """
                ```argdown
                <Argument 1>: Animals suffer.

                (1) Every animal is a being that can suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "animal", "G": "a being that can suffer"}}
                (2) Whatever is a being that can suffer must not be eaten. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "something that must not be eaten"}}
                -- {from: ["1", "2"]} --
                (3) Every animal is a being that must not be eaten. {formalization: "all x.(F(x) -> H(x))"}
                ```
                """,
                """
                ```argdown
                <Argument 1>: Animals suffer.

                (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "animal", "G": "a being that can suffer"}}
                (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "something that must not be eaten"}}
                -- {from: ["1", "2"]} --
                (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
                ```
                """,                
            ),
        ],
    )
    async def test_preference_pair_generator(
        self,
        example_problem,
        solution_class,
        judge_class,
        source_texts,
        PPG,
        chosen,
        rejected
    ):
        problem = example_problem

        judge = judge_class()
        ppg = PPG()

        am_c = textwrap.dedent(chosen)
        am_r = textwrap.dedent(rejected)

        candidate_solutions = [
            solution_class(argdown_snippet=am_c),
            solution_class(argdown_snippet=am_r),
        ]
        evaluations = await judge.arun(problem, candidate_solutions)
        print(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert am_c in cpps[0]["chosen"][-1]["content"]
        assert am_r in cpps[0]["rejected"][-1]["content"]


@pytest.mark.asyncio
class TestInfRecoFromArgannoFailureTypePreferencePairGenerator:

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            (
                """
                ```argdown
                <Argument 1>

                (1) Animals suffer. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
                (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"H": "eating it is wrong"}}
                -- {from: ["1", "2"]} --
                (3) Eating animals is wrong. {formalization: "all x.(F(x) -> H(x))"}
                ```
                """,
                """
                ```argdown
                <Argument 1>

                (1) Animals suffer. {formalization: "for all x:(F(x) -> G(x))", declarations: {"F": "is animal", "G": "can suffer"}}
                (2) Eating what can suffer is wrong. {formalization: "all x.(G(x) -> H(x))", declarations: {"HH": "eating it is wrong"}}
                -- {from: ["1", "2"]} --
                (3) Eating animals is wrong.
                ```
                """,
            ),
        ],
    )
    async def test_preference_pair_generator(
        self,
        problem_class,
        solution_class,
        judge_class,
        source_texts,
        chosen,
        rejected,
        example_problem
    ):
        
        await HirpoTester.test_generic_failure_type_preference_generator(
            problem_class,
            solution_class,
            judge_class,
            source_texts,
            chosen,
            rejected,
            example_problem=example_problem,
        )        
