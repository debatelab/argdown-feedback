import pytest
import textwrap

from argdown_hirpo.base import Evaluation, Feedback
from argdown_hirpo.tasks.core.argmap import (
    ArgumentMap,
    ArgMapProblem,
    ArgMapProblemGenerator,
    ArgMapSolutionGenerator,
    ArgMapJudge,
    ArgMapFeedbackGenerator,
    ConnectednessPreferencePairGenerator,
    MaxArgsPreferencePairGenerator,
    BalancePreferencePairGenerator,
    MaxSupportsPreferencePairGenerator,
    MaxAttacksPreferencePairGenerator,
    MaxDiameterPreferencePairGenerator,
    MinDiameterPreferencePairGenerator,
    DensityPreferencePairGenerator,
    MaxInDegreePreferencePairGenerator,
    MaxOutDegreePreferencePairGenerator,
    MinLeafsPreferencePairGenerator,
    ShortLabelsPreferencePairGenerator,
    DiverseLabelsPreferencePairGenerator,
    ShortClaimsPreferencePairGenerator,
    LongClaimsPreferencePairGenerator,
    ArgumentClaimSizePreferencePairGenerator,
    IndependentWordingPreferencePairGenerator,
    SourceTextProximityPreferencePairGenerator,
)
from tests.hirpo_tester import HirpoTester

from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS


@pytest.fixture
def problem_class():
    return ArgMapProblem


@pytest.fixture
def problem_generator_class():
    return ArgMapProblemGenerator


@pytest.fixture
def solution_class():
    return ArgumentMap


@pytest.fixture
def solution_generator_class():
    return ArgMapSolutionGenerator


@pytest.fixture
def judge_class():
    return ArgMapJudge


@pytest.fixture
def feedback_generator_class():
    return ArgMapFeedbackGenerator


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)
    ]


@pytest.fixture
def valid_argmaps() -> list[ArgumentMap]:
    return [
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
        <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
        ),
    ]


@pytest.fixture
def invalid_argmaps() -> list[ArgumentMap]:
    return [
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate change>: Animal farming causes climate change.
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
          <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
            <+ Animals suffer.
            <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Suffering>: Animal farming causes climate change.
        ```
        """)
        ),
        ArgumentMap(
            argdown_snippet=textwrap.dedent("""
        ```argdown
        [No meat]: We should stop eating meat.
            <+ <Climate change>: Animal farming causes climate change.

        <Suffering>: Animals suffer.

        (1) Animals suffer.
        -----
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
        1. The solution provided does not follow the required formatting convetions Specifically, \
        it the argdown code block is not opened with '```argdown'.
                                 
        **Instructions for Improvement:**
        1. Start the codeblock with '```argdown'.
        """),
    )


def test_avail(model_kwargs):
    llm_available()


@pytest.mark.asyncio
async def test_annotation_problem_generator(
    problem_generator_class,
    problem_class,
    source_texts
):
    await HirpoTester.test_problem_generator(
        problem_generator_class,
        problem_class,
        source_texts
    )


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_argmap_solution_generator(
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
async def test_argmap_judge_valid(
    problem_generator_class,
    judge_class,
    valid_argmaps,
    source_texts
):
    await HirpoTester.test_judge_valid(
        problem_generator_class,
        judge_class,
        valid_argmaps,
        source_texts
    )


@pytest.mark.asyncio
async def test_argmap_judge_invalid(
    problem_generator_class,
    judge_class,
    invalid_argmaps,
    source_texts
):
    await HirpoTester.test_judge_invalid(
        problem_generator_class,
        judge_class,
        invalid_argmaps,
        source_texts
    )


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_feedback_generator(
    problem_generator_class,
    judge_class,
    feedback_generator_class,
    invalid_argmaps,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_feedback_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        invalid_argmaps,
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
    invalid_argmaps,
    source_texts,
    model_kwargs
):
    await HirpoTester.test_revised_solution_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        solution_generator_class,
        solution_class,
        invalid_argmaps,
        source_texts,
        model_kwargs
    )


@pytest.mark.asyncio
class TestArgMapPreferencePairGenerators:
    @pytest.fixture
    def problem(self) -> ArgMapProblem:
        return ArgMapProblem(
            sources=textwrap.dedent("""
        We should not eat meat.
        Animals suffer.
        Farming causes cliamte change.                                             
        """)
        )

    @pytest.fixture
    def judge(self) -> ArgMapJudge:
        return ArgMapJudge()

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
            (
                ConnectednessPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.

            <Suffering>: Animals suffer.

            <Farming>: Farming causes cliamte change.                                             
            ```
            """,
            ),
            (
                MaxArgsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
            ```
            """,
            ),
            (
                BalancePreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                - <Farming>: Farming alleviates climate change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes climate change.
            ```
            """,
            ),
            (
                MaxSupportsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
            ```
            """,
            ),
            (
                MaxAttacksPreferencePairGenerator,
                """
            ```argdown
            [Meat]: We may eat meat.
                - <Suffering>: Animals suffer.
                - <Farming>: Farming causes climate change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes climate change.
            ```
            """,
            ),
            (
                MaxDiameterPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + <Farming>: Farming causes cliamte change.
                        + <Carbon Cycle>: Farming changes the carbon cycle.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
                + <Carbon Cycle>: Farming changes the carbon cycle.
            ```
            """,
            ),
            (
                MaxDiameterPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + [No meat]
                    + <Farming>: Farming causes cliamte change.
                        + <Carbon Cycle>: Farming changes the carbon cycle.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + [No meat]
                + <Farming>: Farming causes cliamte change.
            ```
            """,
            ),
            (
                MinDiameterPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + <Farming>: Farming causes cliamte change.
                        + <Carbon Cycle>: Farming changes the carbon cycle.
            ```
            """,
            ),
            (
                DensityPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
                    +> <Suffering>
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.

            <Farming>: Farming causes cliamte change.
            ```
            """,
            ),
            (
                MaxInDegreePreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + <Farming>: Farming causes cliamte change.
            ```
            """,
            ),
            (
                MaxOutDegreePreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                +> <Eggs>: Eggs ok.
                +> <Milk>: Milk ok.
            ```
            """,
                """
            ```argdown"
            [No meat]: We should not eat meat.
                +> <Eggs>: Eggs ok.
                    +> <Milk>: Milk ok.
            ```
            """,
            ),
            (
                MinLeafsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                    + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
            ),
            (
                ShortLabelsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
                + <Farming>: Farming causes cliamte change.
            ```
            """,
                """
            ```argdown
            [No meat ever]: We should not eat meat.
                + <Suffering badly>: Animals suffer.
                + <Farming so stupid>: Farming causes climate change.
            ```
            """,
            ),
            (
                DiverseLabelsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <No meat now>: Animals suffer.
            ```
            """,
            ),
            (
                ShortClaimsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
            
            [Suffering]: Animals suffer.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat, neither from mammals not from fish.
            
            [Suffering]: Animals, both mammals and fish, suffer.
            ```
            """,
            ),
            (
                LongClaimsPreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat, neither from mammals not from fish.
                + <Suffering>: Animals suffer.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer.
            ```
            """,
            ),
            (
                ArgumentClaimSizePreferencePairGenerator,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer really badly. That is why we should not eat meat.
                + <Farming>: Farming causes climate change. Especially the carbon cycle is affected.
            ```
            """,
                """
            ```argdown
            [No meat]: We should not eat meat.
                + <Suffering>: Animals suffer really badly.
                + <Farming>: Farming causes climate change. Especially the carbon cycle is affected. That is why we should not eat meat. We must not do so at all.
            ```
            """,
            ),
            (
                IndependentWordingPreferencePairGenerator,
                """
            ```argdown
            [A]: It is wrong to consume meat.
                + <B>: Animals are sentient beings.
                + <C>: Farming changes the carbon cycle.                                             
            ```
            """,
                """
            ```argdown
            [A]: We should not eat meat.
                + <B>: Animals suffer.
                + <C>: Farming causes cliamte change.                                             
            ```
            """,
            ),
            (
                SourceTextProximityPreferencePairGenerator,
                """
            ```argdown
            [A]: We should not eat meat.
                + <B>: Animals suffer.
                + <C>: Farming causes cliamte change.                                             
            ```
            """,
                """
            ```argdown
            [A]: It is wrong to consume meat.
                + <B>: Animals are sentient beings.
                + <C>: Farming changes the carbon cycle.                                             
            ```
            """,
            ),
        ],
    )
    async def test_connectedness_preference_pair_generator(
        self, problem, judge, PPG, chosen, rejected
    ):
        ppg = PPG()

        am_c = textwrap.dedent(chosen)
        am_r = textwrap.dedent(rejected)

        candidate_solutions = [
            ArgumentMap(argdown_snippet=am_c),
            ArgumentMap(argdown_snippet=am_r),
        ]
        evaluations = await judge.arun(problem, candidate_solutions)
        print(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert am_c in cpps[0]["chosen"][-1]["content"]
        assert am_r in cpps[0]["rejected"][-1]["content"]
