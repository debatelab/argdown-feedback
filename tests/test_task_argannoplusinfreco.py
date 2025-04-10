from pprint import pprint
import pytest
import textwrap

from argdown_feedback.tasks.base import Feedback, Solution, GenericFeedbackGenerator, GenericSolutionGenerator
from argdown_feedback.tasks.compound.arganno_plus_infreco import (
    ArgannoPlusInfrecoProblem,
    ArgannoPlusInfrecoProblemGenerator,
    ArgannoPlusInfreco,
    ArgannoPlusInfrecoJudge,
    AnnotationProximityPreferencePairGenerator,
)
from argdown_feedback.tasks.core.infreco import (
    NoUnusedPropsPreferencePairGenerator,
    SimplicityPreferencePairGenerator,
)
from argdown_feedback.tasks.core.arganno import (
    AnnotationScopePreferencePairGenerator,
    AnnotationSupportsPreferencePairGenerator,
    AnnotationCoveragePreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgannoPlusInfrecoProblem

@pytest.fixture
def problem_generator_class():
    return ArgannoPlusInfrecoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgannoPlusInfreco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgannoPlusInfrecoJudge

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
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: []}

            <Climate Change>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" argument_label="Suffering" ref_reco_label="2">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1','2']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="5">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
                <+ Animals are screaming.
            (2) If they suffer, they have rights. {annotation_ids: []}
            -- {from: ["1", "2"]} --
            (3) Animals have rights. {annotation_ids: []}
            (4) If they have rights, we should stop eating meat. {annotation_ids: []}
            -- {from: ["3", "4"]} --
            (5) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Bad stuff...
                            
            ```xml
            <proposition argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -----
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```

            Revised stuff:

            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: []}

            <Climate Change>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
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
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            Wrong annotation_id in [No meat]

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}

            <Climate Change>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Free-floating <Climate Change> argument 

            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}

            <Climate Change>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) We should stop eating meat. {annotation_ids: []}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}

            <Climate Change>
                            
            (1) Animals suffer. {annotation_ids: []}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: []}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Empty annotation

            ```xml
            We should stop eating meat.
                            
            Animals suffer. Animal farming causes climate change.
            ```

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2']}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Argmap instead of standard form
                            
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
                ArgannoPlusInfreco.from_raw_answer(
                    textwrap.dedent("""
                    ```xml
                    <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                    ```

                    ```argdown
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2']}
                    -- {from: ["1"]} --
                    (2) [No meat]: We should stop eating meat. {annotation_ids: ['2']}

                    <Climate Change>
                                    
                    (1) Animals suffer. {annotation_ids: []}
                    -- {from: ["1"]} --
                    (2) We should stop eating meat. {annotation_ids: ['1']}
                    ```
                    """)
                ),
                ArgannoPlusInfreco.from_raw_answer(
                    textwrap.dedent("""
                    ```xml
                    <proposition id="1" argument_label="Climate Change" ref_reco_label="2">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                    ```

                    ```argdown
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2']}
                    -- {from: ["1"]} --
                    (2) [No meat]: We should stop eating meat. {annotation_ids: ['2']}

                    <Climate Change>
                                    
                    (1) Animals suffer. {annotation_ids: []}
                    -- {from: ["1"]} --
                    (2) We should stop eating meat. {annotation_ids: ['1']}
                    ```
                                    
                    <No argument>: Nothing
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
class TestArgannoPlusInfrecoPreferencePairGenerators:

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
            (
                AnnotationProximityPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Most living beings feel pain. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: It is wrong to eat meat. {annotation_ids: ['1']}
                ```
                """,
            ),
            # (
            #     NoUnusedPropsPreferencePairGenerator,
            #     """
            #     ```xml
            #     <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
            #     <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            #     ```

            #     ```argdown
            #     <Suffering>
                                
            #     (1) Animals suffer. {annotation_ids: ['2']}
            #     -- {from: ["1"]} --
            #     (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            #     ```
            #     """,
            #     """
            #     ```xml
            #     <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
            #     <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            #     ```

            #     ```argdown
            #     <Suffering>
                                
            #     (1) Animals suffer. {annotation_ids: ['2']}
            #     (2) Another premise, unused. {annotation_ids: []}
            #     -- {from: ["1"]} --
            #     (3) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            #     ```
            #     """,
            # ),
            (
                SimplicityPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer and feel pain. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat and not consume any animal products. {annotation_ids: ['1']}
                ```
                """,
            ),
            (
                AnnotationScopePreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                Animals suffer. Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: []}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
            ),
            (
                AnnotationSupportsPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
            ),
            (
                AnnotationCoveragePreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer. Animal farming causes climate change.</proposition>
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
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

        chosen = ArgannoPlusInfreco.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgannoPlusInfreco.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
