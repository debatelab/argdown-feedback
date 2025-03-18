from openai import OpenAI
import pytest
import textwrap
import warnings

from argdown_hirpo.base import Feedback, HIRPreferencePairGenerator
from argdown_hirpo.tasks.core.arganno import(
    Annotation,
    AnnotationProblem,
    AnnotationProblemGenerator,
    AnnotationSolutionGenerator,
    AnnotationJudge,
    AnnotationFeedbackGenerator,
    AnnotationScopePreferencePairGenerator,
    AnnotationSupportsPreferencePairGenerator,
    AnnotationAttacksPreferencePairGenerator,
    AnnotationNoAttacksPreferencePairGenerator,
    AnnotationCoveragePreferencePairGenerator,
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
            warnings.warn(UserWarning(f"Model {model_id} not available at local inference server {base_url} (available models are: {[model.id for model in models.data]})"))
        return avail
    except Exception as e:
        warnings.warn(UserWarning(f"Could not connect to local inference server {base_url} (Error: {e})"))
        return False


@pytest.fixture
def source_texts() -> list[str]:
    return [
        textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """),
        textwrap.dedent("""
        We should continue eating meat.
        It is a tradition. It is healthy.
        My mum cooks it.
        """),
    ]

@pytest.fixture
def valid_annotations1() -> list[Annotation]:
    return [
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2" attacks="">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> <proposition id="3" supports="1 2">Animal farming causes climate change.</proposition>
        ```
        """)),
    ]


@pytest.fixture
def invalid_annotations1() -> list[Annotation]:
    return [
        Annotation(annotated_source_text=textwrap.dedent("""
        ```
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal farming causes climate change.
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        You should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should <proposition id="1a">stop eating meat.</proposition></proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition>We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="3">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2" attacks="3">Animals suffer.</proposition> Animal farming causes climate change.
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> <proposition id="3" from="1 2">Animal farming causes climate change.</proposition>
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <claim id="2">Animals suffer.</claim> Animal farming causes climate change.
        ```
        """)),
    ]

@pytest.fixture
def feedback1() -> Feedback:
    return Feedback(
        prompt="Please provide feedback.",
        feedback=textwrap.dedent("""
        **Feedback:**
        1. The solution provided does not follow the required annotation scheme. Specifically, \
        it uses an unknown element 'claim' which is not part of the specified XML elements.
        2. There's a lack of argumentative structure in the annotations, failing to connect \
        propositions with each other or clarify their roles within the argument.
        3. No explicit claims have been identified and properly tagged as premises or conclusions.
                                 
        **Instructions for Improvement:**
        1. **Understand the Task:** Before proceeding, ensure you comprehend the given XML schema \
        and the purpose of annotating the source text according to it. Familiarize yourself with \
        the elements such as `<proposition>` and their attributes like `id`, `supports`, and `attacks`.
        2. **Annotate Key Claims:** Identify the central claims within the text that support or \
        refute each other. Use the `<proposition>` element to annotate these claims, assigning them \
        unique IDs for reference.
        3. **Clarify Relationships:** Determine which propositions are premises (supporting reasons) \
        and which ones are conclusions (conclusions). Use attributes like `supports` and `attacks` \
        to indicate how different propositions relate to each other in terms of support or attack.
        4. **Correct Unknown Elements:** When annotating, stick strictly to the allowed elements and \
        attributes provided by the schema. The `<proposition>` element is the only one you should \
        use for marking argumentative components.
        5. **Example of Corrected Annotation:** Here's a simplified example of how a part of the \
        source text might be annotated correctly:
                                 
        ```
        <proposition id=\"1\" ref_reco_label=\"A\">We should stop eating meat.</proposition>\
        <proposition id=\"2\" supports=\"1\">Animals suffer. Animal farming causes climate change.</proposition>
        ```
        
        6. **Thoroughly Review the Text:** Analyze the text to find explicit claims and implicit premises \
        that could be used as support or counterarguments for other propositions.
        7. **Use External Resources If Necessary:** If you're struggling to understand parts of the argument, \
        consult external resources like a detailed summary or an explanation of the argument's key points \
        provided in your assignment materials.
        8. **Double-Check Your Work:** After completing the annotations, review them carefully to ensure they \
        accurately reflect the argumentative structure and relationships within the text.
                                 
        By following these steps and adhering strictly to the annotation scheme, you should be able to create\
        a valid and informative annotation of the source text that represents its argumentative content \
        effectively.
        """),
    )

def hirp_factory(model_kwargs, vpp_gen=AnnotationSupportsPreferencePairGenerator):
    return HIRPreferencePairGenerator(
        problem_generator=AnnotationProblemGenerator(),
        solution_generator=AnnotationSolutionGenerator(n_solutions=8, **model_kwargs),
        judge=AnnotationJudge(),
        feedback_generator=AnnotationFeedbackGenerator(**model_kwargs),
        virtue_preference_pair_generator=vpp_gen(),
    )


def test_avail(model_kwargs):
    llm_available()


@pytest.mark.asyncio
async def test_annotation_problem_generator(source_texts):
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_texts)
    assert isinstance(problem, AnnotationProblem)

    print(problem.instruct_prompt())
    print(problem.revise_prompt())

    assert source_texts[0] in problem.instruct_prompt()
    assert source_texts[1] in problem.instruct_prompt()
    assert source_texts[0] in problem.instruct_prompt(ask_for_invalid=True)
    assert source_texts[1] in problem.instruct_prompt(ask_for_invalid=True)

    assert "!WARNING" in problem.instruct_prompt(ask_for_invalid=True)
    assert "!WARNING" in problem.revise_prompt(ask_for_invalid=True)


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_annotation_solution_generator(source_texts, model_kwargs):
    pg = AnnotationProblemGenerator()
    sg = AnnotationSolutionGenerator(n_solutions=1, **model_kwargs)  # lmstudio server does not support param n
    problem = await pg.arun(source_texts)
    solutions = await sg.arun(problem)
    assert len(solutions) == 1
    for i, sol in enumerate(solutions):
        print(f"## Annotation {i+1}")
        print(sol)
        assert isinstance(sol, Annotation)


@pytest.mark.asyncio
async def test_annotation_judge_valid(valid_annotations1, source_texts):
    source_text = source_texts[0]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, valid_annotations1)
    assert len(evaluations) == len(valid_annotations1)
    for i, ev in enumerate(evaluations):
        print(f"## Annotation {i+1}")
        print(ev)
        assert ev.is_valid
        assert not any(v for _, v in ev.artifacts["eval_metrics"].items())
        assert ev.artifacts["soup"]



@pytest.mark.asyncio
async def test_annotation_judge_invalid(invalid_annotations1, source_texts):
    source_text = source_texts[0]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, invalid_annotations1)
    assert len(evaluations) == len(invalid_annotations1)
    for i, ev in enumerate(evaluations):
        print(f"## Annotation {i+1}")
        print(ev)
        assert not ev.is_valid
        assert any(v for _, v in ev.artifacts["eval_metrics"].items())
        assert ev.artifacts["soup"]


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_feedback_generator(invalid_annotations1, source_texts, model_kwargs):

    source_text = source_texts[0]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, invalid_annotations1)

    fg = AnnotationFeedbackGenerator(n_feedbacks=1, **model_kwargs)
    for annotation, evaluation in zip(invalid_annotations1, evaluations):
        feedbacks = await fg.arun(problem, annotation, evaluation)
        assert len(feedbacks) == 1
        feedback = feedbacks[0]
        assert isinstance(feedback, Feedback)
        assert problem.instruct_prompt() in feedback.prompt
        assert str(annotation) in feedback.prompt
        print(feedback)


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_revised_solution_generator(invalid_annotations1, source_texts, model_kwargs):
    source_text = source_texts[0]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)
    annotation = invalid_annotations1[-1]

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, [annotation])
    evaluation = evaluations[0]

    fg = AnnotationFeedbackGenerator(n_feedbacks=1, **model_kwargs)
    feedbacks = await fg.arun(problem, annotation, evaluation)
    feedback = feedbacks[0]

    sg = AnnotationSolutionGenerator(n_solutions=1, **model_kwargs)  # lmstudio server does not support param n
    revised_annotations = await sg.arun(problem=problem, original_solution=annotation, feedback=feedback)
    assert len(revised_annotations) == 1
    revised_annotation = revised_annotations[0]
    assert isinstance(revised_annotation, Annotation)
    print(revised_annotation)
