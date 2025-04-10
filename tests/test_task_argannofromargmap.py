import pytest
import textwrap

from argdown_feedback.tasks.base import Feedback, Solution, GenericSolutionGenerator
from argdown_feedback.tasks.core.argmap import (
    ArgMapJudge,
    ArgMapProblem,
    ArgumentMap,
)
from argdown_feedback.tasks.core.arganno import (
    Annotation,
    AnnotationJudge,
    AnnotationFeedbackGenerator,
    AnnotationCoveragePreferencePairGenerator
)
from argdown_feedback.tasks.sequential.arganno_from_argmap import (
    ArgannoFromArgmapProblem,
    ArgannoFromArgmapProblemGenerator,
    ArgmapTextProximityPreferencePairGenerator,
    ArgmapGraphProximityPreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgannoFromArgmapProblem

@pytest.fixture
def problem_generator_class():
    return ArgannoFromArgmapProblemGenerator

@pytest.fixture
def solution_class():
    return Annotation

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return AnnotationJudge

@pytest.fixture
def feedback_generator_class():
    return AnnotationFeedbackGenerator


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)
    ]


@pytest.fixture
def example_problem() -> ArgannoFromArgmapProblem:
        sources = textwrap.dedent("""
            We should stop eating meat.
                            
            Animals suffer. Animal farming causes climate change.
            """)
        argdown_snippet = textwrap.dedent("""
            ```argdown
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate change>: Animal farming causes climate change.
            ```
            """)
        argmap_evaluation = ArgMapJudge()._evaluate_argmap(ArgMapProblem(argdown_snippet), ArgumentMap(argdown_snippet))
        return ArgannoFromArgmapProblem(
            sources=sources,
            argdown_snippet=argdown_snippet,
            argdown_map=argmap_evaluation.artifacts.get("argdown_map"),
            argmap_evaluation=argmap_evaluation,
        ) 



@pytest.fixture
def valid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            annotated_source_text=textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class(
            annotated_source_text=textwrap.dedent("""
            ```
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```
            """)
        ),
        solution_class(
            annotated_source_text=textwrap.dedent("""
            No id.
                                            
            ```xml
            <proposition argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
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
        the annotation code block is not opened with '```xml'.
                                 
        **Instructions for Improvement:**
        1. Start the codeblock with '```xml'.
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
        argdown_artifact_keys=[],
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
        argdown_artifact_keys=[],
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
                AnnotationCoveragePreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop</proposition> eating meat.
                                
                Animals <proposition id="2" supports="1" argument_label="Suffering">suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming</proposition> causes climate change.
                ```
                """,
            ),
            (
                ArgmapTextProximityPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop</proposition> eating meat.
                                
                Animals <proposition id="2" supports="1" argument_label="Suffering">suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming</proposition> causes climate change.
                ```
                """,
            ),
            (
                ArgmapGraphProximityPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
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
            solution_class(annotated_source_text=am_c),
            solution_class(annotated_source_text=am_r),
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
                ```xml
                <proposition argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```
                """,
                """
                ```xml
                <proposition argument_label="No meat">We should stop eating meat.</proposition> CHANGED THE SOURCE TREXT
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
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
