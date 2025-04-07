from pprint import pprint
import pytest
import textwrap

from argdown_hirpo.tasks.base import (
    Feedback,
    Solution,
    GenericFeedbackGenerator,
    GenericSolutionGenerator,
)
from argdown_hirpo.tasks.compound.arganno_plus_logreco import (
    ArgannoPlusLogRecoProblem,
    ArgannoPlusLogRecoProblemGenerator,
    ArgannoPlusLogReco,
    ArgannoPlusLogRecoJudge,
)
from argdown_hirpo.tasks.compound.arganno_plus_infreco import (
    AnnotationProximityPreferencePairGenerator,
)
from argdown_hirpo.tasks.core.infreco import (
    SimplicityPreferencePairGenerator,
)
from argdown_hirpo.tasks.core.arganno import (
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
    return ArgannoPlusLogRecoProblem

@pytest.fixture
def problem_generator_class():
    return ArgannoPlusLogRecoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgannoPlusLogReco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgannoPlusLogRecoJudge

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
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
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
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {"p": "Animals suffer."}}
                <+ Animals are screaming.
            (2) If they suffer, they have rights. {annotation_ids: [], formalization: "p -> q", declarations: {q: "Animals have rights."}}
            -- {from: ["1", "2"]} --
            (3) Animals have rights. {annotation_ids: [], formalization: "q"}
            (4) If they have rights, we should stop eating meat. {annotation_ids: [], formalization: "q -> r", declarations: {"r": "No meat."}}
            -- {from: ["3", "4"]} --
            (5) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "r"}
            ```
            """)
        ),    ]


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
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            Non sequitur

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p | q", declarations: {"p": "Animals suffer.", q: "Very much."}}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            Inconsistent premises

            ```argdown
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & -p", declarations: {"p": "Animals suffer."}}
            -- {from: ["1"]} --
            (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""
            ```xml
            <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
            ```

            Missing formalization

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
            Wrong argument_labels

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
class TestArgannoPlusLogRecoFailureTypePreferencePairGenerator:

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            (
                ArgannoPlusLogReco.from_raw_answer(
                    textwrap.dedent("""
                    ```xml
                    <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Climate Change" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                    ```

                    ```argdown
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                    -- {from: ["1"]} --
                    (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                    ```
                    """)
                ),
                ArgannoPlusLogReco.from_raw_answer(
                    textwrap.dedent("""
                    ```xml
                    <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Climate Change" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                    ```

                    ```argdown
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2'], formalization: "p | q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                    -- {from: ["1"]} --
                    (2) [No meat]: We should stop eating meat. {annotation_ids: ['3'], formalization: "p"}
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
class TestArgannoPlusLogRecoPreferencePairGenerators:

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
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: Animals must not be consumed. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
            ),
            (
                SimplicityPreferencePairGenerator,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer and feel pain. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat and not consume animal products. {annotation_ids: ['1'], formalization: "p"}
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
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                Animals suffer. Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: [], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
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
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
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
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
                ```
                """,
                """
                ```xml
                <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
                ```

                ```argdown
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
                -- {from: ["1"]} --
                (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
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

        chosen = ArgannoPlusLogReco.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgannoPlusLogReco.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
