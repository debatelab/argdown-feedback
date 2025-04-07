from pprint import pprint
import pytest
import textwrap

from argdown_hirpo.tasks.base import Feedback, Solution, GenericSolutionGenerator
from argdown_hirpo.tasks.core.arganno import AnnotationJudge
from argdown_hirpo.tasks.core.argmap import (
    ArgumentMap,
    ArgMapJudge,
    ArgMapFeedbackGenerator,
    MaxArgsPreferencePairGenerator
)
from argdown_hirpo.tasks.sequential.argmap_from_arganno import (
    ArgmapFromArgannoProblem,
    ArgmapFromArgannoProblemGenerator,
    AnnotationTextProximityPreferencePairGenerator,
    AnnotationGraphProximityPreferencePairGenerator
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgmapFromArgannoProblem

@pytest.fixture
def problem_generator_class():
    return ArgmapFromArgannoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgumentMap

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

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
def example_problem() -> ArgmapFromArgannoProblem:
        xml_snippet = textwrap.dedent("""
            ```xml
            <proposition id="1">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1">Animals suffer.</proposition> <proposition id="3" supports="2">Animal farming causes climate change.</proposition>
            ```
            """)
        soup_anno, _ = AnnotationJudge().parse_xml_snippet(xml_snippet)
        return ArgmapFromArgannoProblem(
            annotated_text=xml_snippet,
            soup_anno=soup_anno,
        ) 


@pytest.fixture
def valid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            argdown_snippet=textwrap.dedent("""
            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                + <Suffering>: Animals suffer. {annotation_ids: ['2']}
              + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
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
                AnnotationTextProximityPreferencePairGenerator,
                """
                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                ```argdown
                [No meat]: Wir sollten keine Tiere essen. {annotation_ids: ['1']}
                    + <Suffering>: Tiere leiden. {annotation_ids: ['2']}
                    + <Climate change>: Tierhaltung verursacht THG-Emissionen. {annotation_ids: ['3']}
                ```
                """,
            ),
            (
                AnnotationGraphProximityPreferencePairGenerator,
                """
                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                ```argdown
                [No meat]: We should stop eating meat.
                    + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
            ),
            (
                MaxArgsPreferencePairGenerator,
                """
                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                    - <Health>: Eating meat is healthy.
                ```
                """,
                """
                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    + <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    + <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
            )
        ],
    )
    async def test_preference_pair_generator(
        self,
        example_problem,
        solution_class,
        judge_class,
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
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        pprint(cpps)
        assert len(cpps) == 1
        assert am_c in cpps[0]["chosen"][-1]["content"]
        assert am_r in cpps[0]["rejected"][-1]["content"]


@pytest.mark.asyncio
class TestArgmapFromArgannoFailureTypePreferencePairGenerator:

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            (
                """
                ```argdown
                We should stop eating meat.
                    + <Suffering>: Animals suffer.
                    + <Climate change>: Animal farming causes climate change.
                ```
                """,
                """
                ```argdown
                We should stop eating meat.
                    + <Suffering>: Animals suffer.
                    + <Climate change>: Animal farming causes climate change.

                <Climate change>

                (1) Animal farming causes climate change.
                -----
                (2) We should be vegans.
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
        example_problem,
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
