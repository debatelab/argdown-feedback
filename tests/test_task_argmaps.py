from openai import OpenAI
import pytest
import textwrap
import warnings

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


MODEL_KWARGS = {
    "inference_base_url": "http://localhost:8000/v1",
    "model_id": "debatelabkit/llama-3.1-argunaut-1-8b-spin-gguf/llama-3.1-argunaut-1-8b-spin-q4_k_m.gguf",
}


@pytest.fixture
def model_kwargs():
    return MODEL_KWARGS


def llm_available() -> bool:
    base_url = MODEL_KWARGS["inference_base_url"]
    model_id = MODEL_KWARGS["model_id"]
    try:
        models = OpenAI(api_key="EMPTY", base_url=base_url).models.list()
        avail = model_id in [model.id for model in models.data]
        if not avail:
            warnings.warn(
                UserWarning(
                    f"Model {model_id} not available at local inference server {base_url} (available models are: {[model.id for model in models.data]})"
                )
            )
        return avail
    except Exception as e:
        warnings.warn(
            UserWarning(
                f"Could not connect to local inference server {base_url} (Error: {e})"
            )
        )
        return False


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
async def test_annotation_problem_generator(source_texts):
    pg = ArgMapProblemGenerator()
    problem = await pg.arun(source_texts)
    assert isinstance(problem, ArgMapProblem)

    print(problem.instruct_prompt())
    print(problem.revise_prompt())

    assert source_texts[0] in problem.instruct_prompt()
    assert source_texts[0] in problem.instruct_prompt(ask_for_invalid=True)
    assert "super cool hint" in problem.instruct_prompt(hints=["super cool hint"])

    assert "!WARNING" in problem.instruct_prompt(ask_for_invalid=True)
    assert "!WARNING" in problem.revise_prompt(ask_for_invalid=True)

    inv_prompt = problem.instruct_prompt(ask_for_invalid=True, evaluation=Evaluation(is_valid=False, artifacts={}, metrics={"level": "AAA"}))
    assert "!WARNING" in inv_prompt
    assert "level" in inv_prompt
    assert "AAA" in inv_prompt

    inv_prompt = problem.revise_prompt(ask_for_invalid=True, evaluation=Evaluation(is_valid=False, artifacts={}, metrics={"level": "AAA"}))
    assert "!WARNING" in inv_prompt
    assert "level" in inv_prompt
    assert "AAA" in inv_prompt



@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_argmap_solution_generator(source_texts, model_kwargs):
    pg = ArgMapProblemGenerator()
    sg = ArgMapSolutionGenerator(
        n_solutions=1, **model_kwargs
    )  # lmstudio server does not support param n
    problem = await pg.arun(source_texts)
    solutions = await sg.arun(problem)
    assert len(solutions) == 1
    for i, sol in enumerate(solutions):
        print(f"## Argmap {i + 1}")
        print(sol)
        assert isinstance(sol, ArgumentMap)


@pytest.mark.asyncio
async def test_argmap_judge_valid(valid_argmaps, source_texts):
    source_text = source_texts[0]
    pg = ArgMapProblemGenerator()
    problem = await pg.arun(source_text)

    judge = ArgMapJudge()
    evaluations = await judge.arun(problem, valid_argmaps)
    assert len(evaluations) == len(valid_argmaps)
    for i, ev in enumerate(evaluations):
        print(f"## ArgMap {i + 1}")
        print(ev)
        assert ev.is_valid
        assert not any(v for _, v in ev.metrics.items())
        assert ev.artifacts["argdown"]


@pytest.mark.asyncio
async def test_argmap_judge_invalid(invalid_argmaps, source_texts):
    source_text = source_texts[0]
    pg = ArgMapProblemGenerator()
    problem = await pg.arun(source_text)

    judge = ArgMapJudge()
    evaluations = await judge.arun(problem, invalid_argmaps)
    assert len(evaluations) == len(invalid_argmaps)
    for i, ev in enumerate(evaluations):
        print(f"## ArgMap {i + 1}")
        print(ev)
        argdown = ev.artifacts.get("argdown")
        if argdown:
            print(argdown.propositions)
            print(argdown.arguments)
        assert not ev.is_valid
        assert any(v for _, v in ev.metrics.items())


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_feedback_generator(invalid_argmaps, source_texts, model_kwargs):
    source_text = source_texts[0]
    pg = ArgMapProblemGenerator()
    problem = await pg.arun(source_text)

    judge = ArgMapJudge()
    evaluations = await judge.arun(problem, invalid_argmaps)

    fg = ArgMapFeedbackGenerator(n_feedbacks=1, **model_kwargs)
    for argmap, evaluation in zip(invalid_argmaps, evaluations):
        feedbacks = await fg.arun(problem, argmap, evaluation)
        assert len(feedbacks) == 1
        feedback = feedbacks[0]
        assert isinstance(feedback, Feedback)
        assert problem.instruct_prompt() in feedback.prompt
        assert str(argmap) in feedback.prompt
        print(feedback)


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_revised_solution_generator(invalid_argmaps, source_texts, model_kwargs):
    source_text = source_texts[0]
    pg = ArgMapProblemGenerator()
    problem = await pg.arun(source_text)
    argmap = invalid_argmaps[-1]

    judge = ArgMapJudge()
    evaluations = await judge.arun(problem, [argmap])
    evaluation = evaluations[0]

    fg = ArgMapFeedbackGenerator(n_feedbacks=1, **model_kwargs)
    feedbacks = await fg.arun(problem, argmap, evaluation)
    feedback = feedbacks[0]

    sg = ArgMapSolutionGenerator(
        n_solutions=1, **model_kwargs
    )  # lmstudio server does not support param n
    revised_argmaps = await sg.arun(
        problem=problem, original_solution=argmap, feedback=feedback
    )
    assert len(revised_argmaps) == 1
    revised_argmap = revised_argmaps[0]
    print(revised_argmap)
    assert isinstance(revised_argmap, ArgumentMap)


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
