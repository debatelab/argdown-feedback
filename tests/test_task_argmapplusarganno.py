from pprint import pprint
import pytest
import textwrap

from argdown_hirpo.base import Feedback, Solution, GenericFeedbackGenerator
from argdown_hirpo.tasks.compound.argmap_plus_arganno import (
    ArgmapPlusArgannoProblem,
    ArgmapPlusArgannoProblemGenerator,
    ArgmapPlusArganno,
    ArgmapPlusArgannoSolutionGenerator,
    ArgmapPlusArgannoJudge,
    AnnotationProximityPreferencePairGenerator,
)
from argdown_hirpo.tasks.core.argmap import (
    ConnectednessPreferencePairGenerator,
    MaxArgsPreferencePairGenerator,
)
from argdown_hirpo.tasks.core.arganno import (
    AnnotationScopePreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgmapPlusArgannoProblem

@pytest.fixture
def problem_generator_class():
    return ArgmapPlusArgannoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgmapPlusArganno

@pytest.fixture
def solution_generator_class():
    return ArgmapPlusArgannoSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgmapPlusArgannoJudge

@pytest.fixture
def feedback_generator_class():
    return GenericFeedbackGenerator


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)
    ]


@pytest.fixture
def valid_recos(solution_class) -> list[Solution]:
    return [
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Bad stuff:
            ```xml
            <proposition id="2" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```

            Good stuff:
                            
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports=["1"] argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports=["1"] argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate change>: Animal farming causes climate change.
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Missing inline yaml
                            
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                <+ <Suffering>: Animals suffer.
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Wrong argument_label in prop 3.
                            
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Wrong annotation_ids in <Suffering>

            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
            ```

            ```argdown
            [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                <+ <Suffering>: Animals suffer. {annotation_ids: ['3']}
                <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
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


@pytest.mark.asyncio
async def test_problem_generator(
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
    source_texts
):
    await HirpoTester.test_judge_valid(
        problem_generator_class,
        judge_class,
        valid_recos,
        source_texts
    )


@pytest.mark.asyncio
async def test_judge_invalid(
    problem_generator_class,
    judge_class,
    invalid_recos,
    source_texts
):
    await HirpoTester.test_judge_invalid(
        problem_generator_class,
        judge_class,
        invalid_recos,
        source_texts
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
class TestInfRecoFromArgannoFailureTypePreferencePairGenerator:

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            (
                ArgmapPlusArganno.from_raw_answer(
                    textwrap.dedent("""
                    Missing inline yaml
                                    
                    ```xml
                    <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                    ```

                    ```argdown
                    [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                    ```
                    """)
                ),
                ArgmapPlusArganno.from_raw_answer(
                    textwrap.dedent("""
                    Missing inline yaml and support mismatch
                                    
                    ```xml
                    <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="2" argument_label="Climate change">Animal farming causes climate change.</proposition>
                    ```

                    ```argdown
                    [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                    ```
                    """)
                ),
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
    ):
        
        await HirpoTester.test_generic_failure_type_preference_generator(
            problem_class,
            solution_class,
            judge_class,
            source_texts,
            chosen,
            rejected,
        )        


@pytest.mark.asyncio
class TestArgmapPlusArgannoPreferencePairGenerators:

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
            (
                ConnectednessPreferencePairGenerator,
                """
                1 component
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                2 components
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}

                <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
            ),
            (
                MaxArgsPreferencePairGenerator,
                """
                2 arguments
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                1 argument
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ [Suffering]: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
            ),
            (
                AnnotationScopePreferencePairGenerator,
                """
                both lines fully covered
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                1 line fully covered
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                Animals suffer. Animal farming causes climate change.
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                //    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                //    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
            ),
            (
                AnnotationProximityPreferencePairGenerator,
                """
                near verbatim identical
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
                ```
                """,
                """
                strong rephrase of 1 claim
                ```xml
                <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
                ```

                ```argdown
                [No meat]: Consuming animals is wrong. {annotation_ids: ['1']}
                    <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
                    <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
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
        problem = problem_class(sources=source_texts[0])

        judge = judge_class()
        ppg = PPG()

        chosen = ArgmapPlusArganno.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgmapPlusArganno.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]

