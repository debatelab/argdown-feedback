from pprint import pprint
import pytest
import textwrap

from argdown_hirpo.base import Feedback, Solution, GenericFeedbackGenerator, GenericSolutionGenerator
from argdown_hirpo.tasks.compound.argmap_plus_infreco import (
    ArgmapPlusInfrecoProblem,
    ArgmapPlusInfrecoProblemGenerator,
    ArgmapPlusInfreco,
    ArgmapPlusInfrecoJudge,
    SimplicityPreferencePairGenerator,
    ConnectednessPreferencePairGeneratorCT,
    MaxArgsPreferencePairGeneratorCT,
    MaxSupportsPreferencePairGeneratorCT,
    MaxAttacksPreferencePairGeneratorCT,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgmapPlusInfrecoProblem

@pytest.fixture
def problem_generator_class():
    return ArgmapPlusInfrecoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgmapPlusInfreco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgmapPlusInfrecoJudge

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
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) We should stop eating meat.
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) NOT: Animals suffer.
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
            
            <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
                <+ Animals are screaming.
            (2) If they suffer, they have rights.
            -- {from: ["1", "2"]} --
            (3) Animals have rights.
            (4) If they have rights, we should stop eating meat.
            -- {from: ["3", "4"]} --
            (5) [No meat]: We should stop eating meat.

            <Climate Change>

            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) Animal farming is bad.                            
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Bad stuff...
                            
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate change>: Animal farming causes climate change.
                <- Unlabeled prop.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.                            
            ```

            Revised stuff:

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown 
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
                <- unlabeled prop
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),        
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.

            Bad conclusion:

            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No no meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            Poor from ref:

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["2"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- {from: ["1"]} --
            (2) [No meat]
            ```
            """)
        ),        
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            No inf info in arg 2:

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer.
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat.
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change.
            -- from: ["1"] --
            (2) [No meat]
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
        source_texts,
        argdown_artifact_keys=["argdown_map", "argdown_reco"]
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
                ArgmapPlusInfreco.from_raw_answer(
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
                ArgmapPlusInfreco.from_raw_answer(
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
class TestArgmapPlusInfrecoPreferencePairGenerators:

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
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
                                
                (1) Most living beings feel pain. {annotation_ids: ['2']}
                -- {from: ["1"]} --
                (2) [No meat]: It is wrong to eat meat. {annotation_ids: ['1']}
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

        chosen = ArgmapPlusInfreco.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgmapPlusInfreco.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
