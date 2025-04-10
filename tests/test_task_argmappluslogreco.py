from pprint import pprint
import pytest
import textwrap

from argdown_feedback.tasks.base import Feedback, Solution, GenericFeedbackGenerator, GenericSolutionGenerator
from argdown_feedback.tasks.compound.argmap_plus_infreco import (
    SimplicityPreferencePairGenerator,
    ConnectednessPreferencePairGeneratorCT,
    MaxArgsPreferencePairGeneratorCT,
    MaxSupportsPreferencePairGeneratorCT,
    MaxAttacksPreferencePairGeneratorCT,
    SourceTextProximityPreferencePairGeneratorCT,
)
from argdown_feedback.tasks.compound.argmap_plus_logreco import (
    ArgmapPlusLogrecoProblem,
    ArgmapPlusLogrecoProblemGenerator,
    ArgmapPlusLogreco,
    ArgmapPlusLogrecoJudge,
    GlobalFormalizationsFaithfulnessPreferencePairGenerator,
)

from .hirpo_tester import HirpoTester
from .util import llm_available, MODEL_KWARGS


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS

@pytest.fixture
def problem_class():
    return ArgmapPlusLogrecoProblem

@pytest.fixture
def problem_generator_class():
    return ArgmapPlusLogrecoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgmapPlusLogreco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgmapPlusLogrecoJudge

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
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
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
            [No meat]: We should stop eating meat. {formalization: "q"}
                            
            [Climate Premise]: Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}

            <Suffering>
                            
            (1) [Suffering premise]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
                            
            <Climate Change>
                            
            (1) [Climate Premise]
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Zombies>: Animals are Zombies.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) [Suffering claim]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Zombies>
                            
            (1) Animals are zombies. {formalization: "r", declarations: {r: "Animals are zombies."}}
            (2) If animals are Zombies, they don't suffer. {formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) Animals do not suffer. {formalization: "-p"}
                >< [Suffering claim]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Zombies>: Animals are Zombies.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) [Suffering claim]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: No more meat. {formalization: "q"}
                            
            <Zombies>
                            
            (1) Animals are zombies. {formalization: "r", declarations: {r: "Animals are zombies."}}
            (2) If animals are Zombies, they don't suffer. {formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) Animals do not suffer. {formalization: "-p"}
                >< [Suffering claim]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Health>: Animals are healthy.
            ```

            ```argdown {filename="reconstructions.ad"}

            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If suffer, then no food. {formalization: "p -> -q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: Animals aren't food. {formalization: "-q"}
                            
            <Health>
                            
            (1) Animals healthy. {formalization: "r", declarations: {r: "Animals healthy."}}
            (2) If animals are healthy, they are food. {formalization: "r -> q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals are food. {formalization: "q"}
                    >< [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Health>: Animals are healthy.
            ```

            ```argdown {filename="reconstructions.ad"}

            // comments are fine

            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                // another comment
            (2) If suffer, then no food. {formalization: "p -> -q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: Animals aren't food. {formalization: "-q"}
                            
            <Health>
                            
            (1) Animals healthy. {formalization: "r", declarations: {r: "Animals healthy."}}
            (2) If animals are healthy, they are food. {formalization: "r -> q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals are food. {formalization: "q"}
                    >< [No meat]
            ```
            """)
        ),
    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Zombies>: Animals are Zombies.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) [Suffering claim]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) No more meat. {formalization: "q"}
                +> [No meat] 
                            
            <Zombies>
                            
            (1) Animals are zombies. {formalization: "r", declarations: {r: "Animals are zombies."}}
            (2) If animals are Zombies, they don't suffer. {formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) Animals do not suffer. {formalization: "-p"}
                >< [Suffering claim]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Health>: Animals are healthy.
            ```

            Axiomatic relations between props not logically grounded.

            ```argdown {filename="reconstructions.ad"}

            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If suffer, then no food. {formalization: "p -> -q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) [No food]: Animals aren't food. {formalization: "-q"}
                            
            <Health>
                            
            (1) Animals healthy. {formalization: "r", declarations: {r: "Animals are zombies."}}
            (2) If animals are healthy, they are food. {formalization: "r -> s", declarations: {s: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals are food. {formalization: "s"}
                    >< [No food]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Health>: Animals are healthy.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If suffer, then no food. {formalization: "p -> -q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals aren't food. {formalization: "-q"}
                    +> [No meat]
                            
            <Health>
                            
            (1) Animals healthy. {formalization: "r", declarations: {r: "Animals healthy."}}
            (2) If animals are healthy, they are food. {formalization: "r -> q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals are food. {formalization: "q"}
                    >< [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Health>: Animals are healthy.
            ```

            ```argdown {filename="reconstructions.ad"}
            [No meat]: We should stop eating meat.                            

            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If suffer, then no food. {formalization: "p -> -q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) [No food]: Animals aren't food. {formalization: "-q"}
                    +> [No meat]
                            
            <Health>
                            
            (1) Animals healthy. {formalization: "r", declarations: {r: "Animals healthy."}}
            (2) If animals are healthy, they are food. {formalization: "r -> q", declarations: {q: "Animals food."}}
            -- {from: ["1","2"]} --
            (3) Animals are food. {formalization: "q"}
                    >< [No food]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Un-labeled claim

            ```argdown {filename="map.ad"}
            We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Illegal map

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
              <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Illegal first reco

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (2) [No meat]: We should stop eating meat. {formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            Not enough arguments

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {formalization: "q"}                            
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
                ArgmapPlusLogreco.from_raw_answer(
                    textwrap.dedent("""
                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    <Suffering>
                                    
                    (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (2) [No meat]: We should stop eating meat. {formalization: "q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]
                    ```
                    """)
                ),
                ArgmapPlusLogreco.from_raw_answer(
                    textwrap.dedent("""
                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.

                    Some totally disallowed sentence.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    <Suffering>
                                    
                    (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (2) [No meat]: We should stop eating meat. {formalization: "q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]
                    ```
                    """)
                ),
            ),
            (
                ArgmapPlusLogreco.from_raw_answer(
                    textwrap.dedent("""
                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    <Suffering>
                                    
                    (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (2) [No meat]: We should stop eating meat. {formalization: "q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]
                    ```
                    """)
                ),
                ArgmapPlusLogreco.from_raw_answer(
                    textwrap.dedent("""
                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    Some totally disallowed sentence.

                    <Suffering>
                                    
                    (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (2) [No meat]: We should stop eating meat. {formalization: "q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]
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
class TestArgmapPlusLogrecoPreferencePairGenerators:

    @pytest.mark.parametrize(
        "PPG,chosen,rejected",
        [
            (
                SimplicityPreferencePairGenerator,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer and feel pain. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer and feel pain, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat, that would be totally wrong. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change via greenhaouse gas emissions. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                ConnectednessPreferencePairGeneratorCT,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.

                <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) No meat.  {formalization: "q"}
                ```
                """,
            ),
            (
                MaxArgsPreferencePairGeneratorCT,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                    <+ <Zombies>: Animals are Zombies.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]

                <Zombies>

                (1) Animals are zombies. {formalization: "s", declarations: {s: "Animals are Zombies."}}
                (2) If animals are Zombies, we should not eat them. {formalization: "s -> q", declarations: {q: "We should not eat animals."}} 
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                MaxSupportsPreferencePairGeneratorCT,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                        <- <Zombies>: Animals are Zombies.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) [Suffering claim]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Zombies>
                                
                (1) Animals are zombies. {formalization: "r", declarations: {r: "Animals are zombies."}}
                (2) If animals are Zombies, they don't suffer. {formalization: "r -> -p", declarations: {p: "Animals suffer."}}
                -- {from: ["1","2"]} --
                (3) Animals do not suffer. {formalization: "-p"}
                    >< [Suffering claim]
                ```
                    """,
            ),
            (
                MaxAttacksPreferencePairGeneratorCT,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                        <- <Zombies>: Animals are Zombies.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) [Suffering claim]: Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Zombies>
                                
                (1) Animals are zombies. {formalization: "r", declarations: {r: "Animals are zombies."}}
                (2) If animals are Zombies, they don't suffer. {formalization: "r -> -p", declarations: {p: "Animals suffer."}}
                -- {from: ["1","2"]} --
                (3) Animals do not suffer. {formalization: "-p"}
                    >< [Suffering claim]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                SourceTextProximityPreferencePairGeneratorCT,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: It is wrong to farm animals for food production and consumption.
                    <+ <Suffering>: All being can feel pain..
                    <+ <Climate Change>: Farming is a majout GHG emissions source.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                GlobalFormalizationsFaithfulnessPreferencePairGenerator,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {formalization: "p", declarations: {p: "Cars are cool."}}
                (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "New York is great."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "I am so happy to be here."}}
                (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "New York is great."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
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

        chosen = ArgmapPlusLogreco.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgmapPlusLogreco.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
