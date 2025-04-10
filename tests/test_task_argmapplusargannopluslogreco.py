from pprint import pprint
import pytest
import textwrap

from argdown_feedback.tasks.base import (
    Feedback,
    Solution,
    GenericFeedbackGenerator,
    GenericSolutionGenerator,
)
from argdown_feedback.tasks.compound.argmap_plus_infreco import (
    SimplicityPreferencePairGenerator,
    ConnectednessPreferencePairGeneratorCT,
    MaxArgsPreferencePairGeneratorCT,
    MaxSupportsPreferencePairGeneratorCT,
    MaxAttacksPreferencePairGeneratorCT,
    SourceTextProximityPreferencePairGeneratorCT,
)
from argdown_feedback.tasks.compound.argmap_plus_logreco import (
    GlobalFormalizationsFaithfulnessPreferencePairGenerator,
)
from argdown_feedback.tasks.compound.argmap_plus_arganno_plus_logreco import (
    ArgmapPlusArgannoPlusLogrecoProblem,
    ArgmapPlusArgannoPlusLogrecoProblemGenerator,
    ArgmapPlusArgannoPlusLogreco,
    ArgmapPlusArgannoPlusLogrecoJudge,
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
    return ArgmapPlusArgannoPlusLogrecoProblem

@pytest.fixture
def problem_generator_class():
    return ArgmapPlusArgannoPlusLogrecoProblemGenerator

@pytest.fixture
def solution_class():
    return ArgmapPlusArgannoPlusLogreco

@pytest.fixture
def solution_generator_class():
    return GenericSolutionGenerator

@pytest.fixture
def judge_class():
    return ArgmapPlusArgannoPlusLogrecoJudge

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
            DEFAULT 
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            Unlike DEFAULT: attacking arg

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
            (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) We should eat animals. {annotation_ids: [], formalization: "q"}
                >< [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            Unlike DEFAULT: objection to supporting arg

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) [AniSuff]: Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, animals don't suffer. {annotation_ids: [], formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) Animals don't suffer. {annotation_ids: [], formalization: "-p"}
                >< [AniSuff]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            With bridge claim in map

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <+ [AniSuff]
                        <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) [AniSuff]: Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, animals don't suffer. {annotation_ids: [], formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) Animals don't suffer. {annotation_ids: [], formalization: "-p"}
                >< [AniSuff]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            AXIOMATIC DRELs OUTSIDE ARGRECOS

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}

            [AniSuff]: Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                >< [AniNoSuff]
                            
                            
            <Suffering>
                            
            (1) [AniSuff]
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, animals don't suffer. {annotation_ids: [], formalization: "r -> -p", declarations: {p: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) [AniNoSuff]: Animals don't suffer. {annotation_ids: [], formalization: "-p"}
            ```
            """)
        ),





    ]


@pytest.fixture
def invalid_recos(solution_class) -> list[Solution]:
    return [
        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            MISSING opening argdown codeblock  
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            MISSING axiomatic relation in logreco

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
            (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) We should eat animals. {annotation_ids: [], formalization: "q"}
            //    >< [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            ILLEGAL REF_RECO IN ANNOPTATION
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="5">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),

        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            ILLEGAL REFERENCE TO PROPOSITION IN ANNOPTATION
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="No meat" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),

        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            EMPTY REFRECO IN ANNOTATION
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            MISMATCH DIALECTICAL RELATIONS MAP<>LOGRECO 
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            MISMATCH DIALECTICAL RELATIONS MAP<>LOGRECO 
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.

            <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                           
            MISMATCH DIALECTICAL RELATIONS MAP<>LOGRECO 
                            
            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                <+ <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}
            <Suffering>
                            
            (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
            -- {from: ["1","2"]} --
            (3) We should stop eating meat now. {annotation_ids: [], formalization: "q"}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""                            
                            
            AXIOMATIC REL NOT GROUNDED

            ```xml {filename="annotation.txt"}
            <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                            
            <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
            ```

            ```argdown {filename="map.ad"}
            [No meat]: We should stop eating meat.
                <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
            ```

            ```argdown {filename="reconstructions.ad"}

            [AniSuff]: Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                >< [AniNoSuff]
                            
                            
            <Suffering>
                            
            (1) [AniSuff]
            (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
            -- {from: ["1","2"]} --
            (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                            
            <Climate Change>
                            
            (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
            (2) If animal farming causes climate change, animals don't suffer. {annotation_ids: [], formalization: "r -> t", declarations: {t: "Animals suffer."}}
            -- {from: ["1","2"]} --
            (3) [AniNoSuff]: Animals don't suffer. {annotation_ids: [], formalization: "t"}
            ```
            """)
        ),
        solution_class.from_raw_answer(
            textwrap.dedent("""          
                GROUNDED DREL NOT IN MAP

                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.

                <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """
            ),
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
                ArgmapPlusArgannoPlusLogreco.from_raw_answer(
                    textwrap.dedent("""                           
                    DEFAULT 
                                    
                    ```xml {filename="annotation.txt"}
                    <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                    ```

                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]
                    ```
                    """)
                ),
                ArgmapPlusArgannoPlusLogreco.from_raw_answer(
                    textwrap.dedent("""                           
                    DEFAULT 
                                    
                    ```xml {filename="annotation.txt"}
                    <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                    
                    <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="2" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                    ```

                    ```argdown {filename="map.ad"}
                    [No meat]: We should stop eating meat.
                        <+ <Suffering>: Animals suffer.
                        <+ <Climate Change>: Animal farming causes climate change.
                    ```

                    ```argdown {filename="reconstructions.ad"}
                    <Suffering>
                                    
                    (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                    (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                    -- {from: ["1","2"]} --
                    (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                                    
                    <Climate Change>
                                    
                    (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                    (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q"}
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
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer and feel pain. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer and feel pain, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. Again: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change and is a major source of GHG emissions. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                ConnectednessPreferencePairGeneratorCT,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.

                <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) No meat. {annotation_ids: [], formalization: "q"}
                ```
                """,
            ),
            (
                MaxArgsPreferencePairGeneratorCT,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="No more meat" ref_reco_label="1">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                <No more meat>: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <No more meat>: We should stop eating meat.

                (1) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q", declarations: {q: "We should not eat animals."}}
                (2) If no meat then vegan. {annotation_ids: [], formalization: "q -> s", declarations: {s: "We should be vegan."}}
                -- {from: ["1","2"]} --
                (3) We should be vegan. {annotation_ids: [], formalization: "s"}

                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                MaxSupportsPreferencePairGeneratorCT,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                                
                <Climate Change>
                                
                (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
                (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) We should eat animals. {annotation_ids: [], formalization: "q"}
                    >< [No meat]
                ```
                """,
            ),
            (
                MaxAttacksPreferencePairGeneratorCT,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                                
                <Climate Change>
                                
                (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
                (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) We should eat animals. {annotation_ids: [], formalization: "q"}
                    >< [No meat]
                ```
                """,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                SourceTextProximityPreferencePairGeneratorCT,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: Wir sollten kein Fleisch essen.
                    <+ <Suffering>: Tiere leiden.
                    <+ <Climate Change>: Massentioerhaltung verursacht Klimawandel.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Tiere leiden. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) Wenn sie leiden, iss sie nicht. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: Ich sollte sie nicht essen. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) MTH verursacht Klimawandel. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) Wenn Klimawnadel, dann bitte nicht mehr essen. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                GlobalFormalizationsFaithfulnessPreferencePairGenerator,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Tiere leiden."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "Wir sollten aufhören sie zu essen."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Tierhaltung führt zu Klimawandel."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "Wir sollten aufhören sie zu essen."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                AnnotationScopePreferencePairGenerator,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                We should stop eating meat.
                                
                <proposition id="2" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: [], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                AnnotationCoveragePreferencePairGenerator,
                """
                DEFAULT 
                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should</proposition> stop eating meat.
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals</proposition> suffer. <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming </proposition>causes climate change.
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
            ),
            (
                AnnotationSupportsPreferencePairGenerator,
                """                                
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <+ <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                                
                <Climate Change>
                                
                (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
                (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]
                ```
                """,
                """
                ```xml {filename="annotation.txt"}
                <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                                
                <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
                ```

                ```argdown {filename="map.ad"}
                [No meat]: We should stop eating meat.
                    <+ <Suffering>: Animals suffer.
                    <- <Climate Change>: Animal farming causes climate change.
                ```

                ```argdown {filename="reconstructions.ad"}
                <Suffering>
                                
                (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
                (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                                
                <Climate Change>
                                
                (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
                (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
                -- {from: ["1","2"]} --
                (3) We should eat animals. {annotation_ids: [], formalization: "q"}
                    >< [No meat]
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

        chosen = ArgmapPlusArgannoPlusLogreco.from_raw_answer(textwrap.dedent(chosen))
        rejected = ArgmapPlusArgannoPlusLogreco.from_raw_answer(textwrap.dedent(rejected))
        candidate_solutions = [chosen, rejected]
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
