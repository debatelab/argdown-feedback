import pytest
import textwrap

from argdown_hirpo.base import Feedback, Solution
from argdown_hirpo.tasks.core.infreco import (
    InformalReco,
    InfRecoSolutionGenerator,
    InfRecoJudge,
    InfRecoFeedbackGenerator,
)
from argdown_hirpo.tasks.sequential.infreco_from_arganno import (
    InfRecoFromArgAnnoProblem,
    InfRecoFromArgAnnoProblemGenerator,
    AnnotationProximityPreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return InfRecoFromArgAnnoProblem

@pytest.fixture
def problem_generator_class():
    return InfRecoFromArgAnnoProblemGenerator

@pytest.fixture
def solution_class():
    return InformalReco

@pytest.fixture
def solution_generator_class():
    return InfRecoSolutionGenerator

@pytest.fixture
def judge_class():
    return InfRecoJudge

@pytest.fixture
def feedback_generator_class():
    return InfRecoFeedbackGenerator


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports=["1"]>Animals suffer.</proposition> <proposition id="3" supports=["2"]>Animal farming causes climate change.</proposition>
        """)
    ]


@pytest.fixture
def valid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument 1>: Animals suffer.

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument 1>: Animals suffer.

            (P1) Animals suffer.
            (P2) Animals suffer.
            -- {from: ["P1", "P2"]} --
            (C1) Animals suffer.
            (P3) Animals suffer.
            -- with Modus ponens {from: ["C1"]} --
            (C2) Eating animals is wrong.
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            <No opening codeblock>: Some gist.

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Invalid argdown syntax.

            (1) Animals suffer.
            -- {from: ["1"]} --
            Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument-a>: Two args.

            (1) Animals have brain.
            -- {from: ["1"]} --
            (2) Animals suffer.

            <Argument-b>: Two args.

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            [Argument]: No argument.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Starts with conclusion

            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Ends with premise

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            (3) Never eat meat.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: With duplicate pcs label.

            (1) Animals suffer.
            -- {from: ["1"]} --
            (1) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument without gist>

            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            (1) No LABEL! Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: With empty from list.

            (1) Animals suffer.
            -- {from: []} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Inf info missing altogether.

            (1) Animals suffer.
            -- from: [] --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Nonexistant ref.
                                            
            (1) Animals suffer.
            -- {from: ["2"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Disallowed material 1.
                                            
            (1) Animals suffer.
                + Animals have brain.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Disallowed material 2.
                                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
                + Animals have brain.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Disallowed material 3.
                                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
                                            
            <Argument> 
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            [Brain]: Animals have brain.
                                            
            <Argument>: Disallowed material 4.
                                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Disallowed material 5.
                                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
                                            
            Not even a sentence is allowed here.
            ```
            """)
        ),
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            <Argument>: Disallowed material 6.
                                            
            (1) Animals suffer. {veracity: "true"}
            -- {from: ["1"]} --
            (2) Eating animals is wrong.
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
    problem_generator_class,
    judge_class,
    valid_recos,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_judge_valid(
        problem_generator_class,
        judge_class,
        valid_recos,
        source_texts,
        model_kwargs
    )


@pytest.mark.asyncio
async def test_judge_invalid(
    problem_generator_class,
    judge_class,
    invalid_recos,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_judge_invalid(
        problem_generator_class,
        judge_class,
        invalid_recos,
        source_texts,
        model_kwargs
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
                AnnotationProximityPreferencePairGenerator,
                """
                ```argdown
                <Suffering>: Animals suffer.

                (1) Animals suffer.
                (2) Animal farming causes climate change.
                -- {from: ["1", "2"]} --
                (3) We should stop eating meat.
                ```
                """,
                """
                ```argdown
                <Suffering>: Animals suffer.
                
                (1) Sentient beings have feelings.
                (2) Meat production is a majour source of greenhouse gas emissions.
                -- {from: ["1", "2"]} --
                (3) It is just wrong to eat meat.
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
        PPG,
        chosen,
        rejected
    ):
        problem = problem_class(annotation=source_texts[0])

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
