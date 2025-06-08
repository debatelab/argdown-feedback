from pprint import pprint
from openai import OpenAI
import pytest
import textwrap
import warnings

from argdown_feedback.tasks.base import Evaluation, Feedback, HIRPOGenStats, HIRPreferencePairGenerator, GenericSolutionGenerator, Solution
from argdown_feedback.tasks.core.arganno import(
    Annotation,
    AnnotationProblem,
    AnnotationProblemGenerator,
    AnnotationJudge,
    AnnotationFeedbackGenerator,
    AnnotationScopePreferencePairGenerator,
    AnnotationSupportsPreferencePairGenerator,
    AnnotationAttacksPreferencePairGenerator,
    AnnotationNoAttacksPreferencePairGenerator,
    AnnotationCoveragePreferencePairGenerator,
)
from tests.hirpo_tester import HirpoTester


MODEL_KWARGS = {
    "inference_base_url": "http://localhost:8000/v1",
    "model_id": "llama-3.2-3b-instruct"
    # "model_id": "debatelabkit/llama-3.1-argunaut-1-8b-spin-gguf/llama-3.1-argunaut-1-8b-spin-q4_k_m.gguf",
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
def problem_class():
    return AnnotationProblem

@pytest.fixture
def problem_generator_class():
    return AnnotationProblemGenerator

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
        """),
        textwrap.dedent("""
        We should continue eating meat.
        It is a tradition. It is healthy.
        My mum cooks it.
        """),
        textwrap.dedent("""
        Pro 1: Killing animals for food is cruel and unethical.
        Raising animals in confinement for slaughter is cruel, and many animals in the United States are not slaughtered humanely.


        Animals are sentient beings that have emotions and social connections. Scientific studies show that cattle, pigs, chickens, and all warm-blooded animals can experience stress, pain, and fear. About 50% of meat produced in the United States comes from confined animal feeding operations (CAFOs), where mistreated animals live in filthy, overcrowded spaces with little or no access to pasture, natural light, or clean air. In CAFOs pigs have their tails cut short; chickens have their toenails, spurs, and beaks clipped; and cows have their horns removed and tails docked with no painkillers. Pregnant pigs are kept in metal gestation crates barely bigger than the pigs themselves. Baby cows raised for veal are tied up and confined in tiny stalls their entire short lives (3-18 weeks). [32][35][41][100][147]

        The Humane Methods of Slaughter Act (HMSA) mandates that livestock be stunned unconscious before slaughter to minimize suffering. However, birds such as chickens and turkey are exempted from the HMS, and a report by the U.S. Government Accountability Organization (GAO) found that the USDA was not “taking consistent actions to enforce the HMSA.” [65][66][90]

        In 2017 (the most recent data available), the United States slaughtered a total of 170.6 million animals for food, including 124.5 million pigs, 33.7 million cows, 9.2 million chickens, and 2.4 million sheep. These animals should not have to die painfully and fearfully to satisfy an unnecessary dietary preference.


        Pro 2: A vegetarian diet is healthful.
        According to the American Dietetic Association, a vegetarian diet can meet protein requirements, provide all the essential amino-acids (the building blocks of protein), and provide all the necessary vitamins, fats, and minerals. And, a vegetarian diet can improve one’s health. [1][2]

        According to the USDA and the Food and Agriculture Organization of the United Nations, meat is not an essential part of a healthy diet. Further, studies have linked heme iron found in red meat with an increased risk of colorectal, stomach, and esophageal cancers. Vegetarian sources of iron like leafy greens and beans contain non-heme iron. [3][4] [68][123][150]

        Meat also has high renal acid levels which the body must neutralize by leaching calcium from the bones, which is then passed into urine and lost. There are many sources of healthy vegetarian calcium including tofu, dark leafy greens like kale, spinach, and collard greens, as well as fortified cereals. [5][128]

        Vegetarian diets can reduce the risk of antibiotic resistance to bacteria, kidney stones, gallstones, death from heart disease, high blood pressure, hypertension, stroke, type 2 diabetes, and cancer. [6][7][8][9][10][40][64][102][122][132][140][148]

        Several studies show that vegetarian diets increase the lifespan of adherents by 3.6 to 7.28 years. [76][86] [121][130]


        Pro 3: A vegetarian diet is better for the environment.
        Overgrazing livestock hurts the environment through soil compaction, erosion, and harm to native plants and animals. Grazing has also damaged streams and riparian areas in the western United States. And, grazing has been a factor in the listing of at least 171 species of animals and plants under the Endangered Species Act because the large tracts of flat land interrupt natural habitats. Abstaining from eating meat would help restore land more naturally suited to provide habitat for native plants and animals. [29][92][93]

        A vegetarian diet also conserves water. Producing one pound of beef takes about 1,800 gallons of water, on pound of pork uses about 576 gallons, one pound of turkey needs about 486 gallons, and each pound of chicken requires about 468 gallons. Meanwhile, a pound of tofu only takes about 302 gallons. [151][152][153]

        Additionally, raising animals for food contributes to air and water pollution. Manure produces toxic hydrogen sulfide and ammonia, which pollute the air and leach poisonous nitrates into nearby waters. Runoff laden with manure is a major cause of “dead zones” in 173,000 miles of U.S. waterways, including the 7,700-square-mile dead zone in the Gulf of Mexico. [32][115][116]

        All told, a vegetarian diet leads to lower greenhouse gas emissions. Greenhouse gases are created by enteric fermentation (aka animal farts and burps), manure decomposition, and deforestation to make room for grazing animals and growing feed. Diets including meat cause the creation of up to 54% more greenhouse gas emissions than vegetarian diets. According to the United Nations Environment Programme, a “worldwide diet change away from animal products” is necessary to stop the worst effects of global climate change. [104][134]
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
def valid_annotations2() -> list[Annotation]:
    return [
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">Pro 1: Killing animals for food is cruel and unethical.</proposition>
        [...]

        <proposition id="2">Pro 2: A vegetarian diet is healthful.</proposition>
        [... Also skipping this ...]

        <proposition id="3">Pro 3: A vegetarian diet is better for the environment.</proposition>
        [...]
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
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        <proposition id="1">We should stop eating meat.</proposition>
                        
        <proposition id="2">Animals suffer.</proposition> Animal [...].
        ```
        """)),
    ]


@pytest.fixture
def invalid_annotations2() -> list[Annotation]:
    return [
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        CHANGED PROP CONTENT
        <proposition id="1">Pro 1: EATING (not "Killing") animals for food is cruel and unethical.</proposition>
        [...]

        <proposition id="2">Pro 2: A vegetarian diet is healthful.</proposition>
        [... Also skipping this ...]

        <proposition id="3">Pro 3: A vegetarian diet is better for the environment.</proposition>
        [...]
        ```
        """)),
        Annotation(annotated_source_text=textwrap.dedent("""
        ```xml
        WRONG ORDER
        <proposition id="2">Pro 2: A vegetarian diet is healthful.</proposition>
        [... Also skipping this ...]

        <proposition id="1">Pro 1: Killing animals for food is cruel and unethical.</proposition>
        [...]

        <proposition id="3">Pro 3: A vegetarian diet is better for the environment.</proposition>
        [...]
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
        solution_generator=GenericSolutionGenerator(n_solutions=1, solution_class=Annotation, **model_kwargs),
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
async def test_annotation_solution_generator(source_texts, model_kwargs):
    pg = AnnotationProblemGenerator()
    sg = GenericSolutionGenerator(solution_class=Annotation, n_solutions=1, **model_kwargs)  # lmstudio server does not support param n
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
        pprint(ev)
        assert ev.is_valid
        assert not any(v for _, v in ev.metrics.items())
        assert ev.artifacts["soup"]


@pytest.mark.asyncio
async def test_annotation_judge_valid2(valid_annotations2, source_texts):
    source_text = source_texts[2]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, valid_annotations2)
    assert len(evaluations) == len(valid_annotations2)
    for i, ev in enumerate(evaluations):
        print(f"## Annotation {i+1}")
        pprint(ev)
        assert ev.is_valid
        assert not any(v for _, v in ev.metrics.items())
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
        assert any(v for _, v in ev.metrics.items())


@pytest.mark.asyncio
async def test_annotation_judge_invalid2(invalid_annotations2, source_texts):
    source_text = source_texts[2]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)

    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, invalid_annotations2)
    assert len(evaluations) == len(invalid_annotations2)
    for i, ev in enumerate(evaluations):
        print(f"## Annotation {i+1}")
        print(ev)
        assert not ev.is_valid
        assert any(v for _, v in ev.metrics.items())


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

    sg = GenericSolutionGenerator(solution_class=Annotation, n_solutions=1, **model_kwargs)  # lmstudio server does not support param n
    revised_annotations = await sg.arun(problem=problem, original_solution=annotation, feedback=feedback)
    assert len(revised_annotations) == 1
    revised_annotation = revised_annotations[0]
    assert isinstance(revised_annotation, Annotation)
    print(revised_annotation)


@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
@pytest.mark.asyncio
async def test_self_critique(model_kwargs, invalid_annotations1, source_texts):
    source_text = source_texts[0]
    invalid_annotations = invalid_annotations1[:2]
    pg = AnnotationProblemGenerator()
    problem = await pg.arun(source_text)
    judge = AnnotationJudge()
    evaluations = await judge.arun(problem, invalid_annotations)


    hirp_generator = hirp_factory(model_kwargs)
    pairs, stats = await hirp_generator.run_self_critique(
        problem=problem,
        candidate_solutions=invalid_annotations,
        evaluations=evaluations
    )
    pprint(pairs)
    pprint(stats)
    assert isinstance(pairs, list)
    assert isinstance(stats, HIRPOGenStats)
    assert stats.n_total == len(pairs)

@pytest.mark.asyncio
class TestAnnotationPreferencePairGenerators:

    async def test_annotation_scope_preference_pair_generator(self):
        judge = AnnotationJudge()
        ppg = AnnotationScopePreferencePairGenerator()
        
        problem = AnnotationProblem(sources="A B C")
        anno01 = '```xml\nA B C\n```'
        anno02 = '```xml\nA <proposition id="1">B</proposition> C\n```'
        anno03 = '```xml\nA <proposition id="1">B</proposition> <proposition id="2">C</proposition>\n```'
        candidate_solutions = [Annotation(annotated_source_text=a) for a in [anno01, anno02, anno03]]
        evaluations = await judge.arun(problem, candidate_solutions)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert anno03 in cpps[0]['chosen'][-1]["content"]
        assert anno01 in cpps[0]['rejected'][-1]["content"] or anno02 in cpps[0]['rejected'][-1]["content"]
        assert anno03 not in cpps[0]['rejected'][-1]["content"]

    async def test_annotation_supports_preference_pair_generator(self):
        judge = AnnotationJudge()
        ppg = AnnotationSupportsPreferencePairGenerator()
        
        problem = AnnotationProblem(sources="A B C")
        anno01 = '```xml\nA <proposition id="1">B</proposition> C\n```'
        anno02 = '```xml\nA <proposition id="1" supports="2">B</proposition> <proposition id="2">C</proposition>\n```'
        anno03 = '```xml\nA <proposition id="1" attacks="2">B</proposition> <proposition id="2">C</proposition>\n```'
        candidate_solutions = [Annotation(annotated_source_text=a) for a in [anno01, anno02, anno03]]
        evaluations = await judge.arun(problem, candidate_solutions)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert anno02 in cpps[0]['chosen'][-1]["content"]
        assert anno02 not in cpps[0]['rejected'][-1]["content"]

    async def test_annotation_attacks_preference_pair_generator(self):
        judge = AnnotationJudge()
        ppg = AnnotationAttacksPreferencePairGenerator()
        
        problem = AnnotationProblem(sources="A B C")
        anno01 = '```xml\nA <proposition id="1">B</proposition> C\n```'
        anno02 = '```xml\nA <proposition id="1" supports="2">B</proposition> <proposition id="2">C</proposition>\n```'
        anno03 = '```xml\nA <proposition id="1" attacks="2">B</proposition> <proposition id="2">C</proposition>\n```'
        candidate_solutions = [Annotation(annotated_source_text=a) for a in [anno01, anno02, anno03]]
        evaluations = await judge.arun(problem, candidate_solutions)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert anno03 in cpps[0]['chosen'][-1]["content"]
        assert anno03 not in cpps[0]['rejected'][-1]["content"]

    async def test_annotation_noattacks_preference_pair_generator(self):
        judge = AnnotationJudge()
        ppg = AnnotationNoAttacksPreferencePairGenerator()
        
        problem = AnnotationProblem(sources="A B C")
        anno01 = '```xml\nA <proposition id="1">B</proposition> C\n```'
        anno02 = '```xml\nA <proposition id="1" supports="2">B</proposition> <proposition id="2">C</proposition>\n```'
        anno03 = '```xml\nA <proposition id="1" attacks="2">B</proposition> <proposition id="2">C</proposition>\n```'
        candidate_solutions = [Annotation(annotated_source_text=a) for a in [anno01, anno02, anno03]]
        evaluations = await judge.arun(problem, candidate_solutions)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert anno03 not in cpps[0]['chosen'][-1]["content"]
        assert anno03 in cpps[0]['rejected'][-1]["content"]

    async def test_annotation_coverage_preference_pair_generator(self):
        judge = AnnotationJudge()
        ppg = AnnotationCoveragePreferencePairGenerator()
        
        problem = AnnotationProblem(sources="A B C")
        anno01 = '```xml\nA <proposition id="1">B</proposition> C\n```'
        anno02 = '```xml\n<proposition id="1" supports="2">A B</proposition> <proposition id="2">C</proposition>\n```'
        anno03 = '```xml\nA <proposition id="1" attacks="2">B</proposition> <proposition id="2">C</proposition>\n```'
        candidate_solutions = [Annotation(annotated_source_text=a) for a in [anno01, anno02, anno03]]
        evaluations = await judge.arun(problem, candidate_solutions)
        assert len([e for e in evaluations if e.is_valid]) == len(candidate_solutions)

        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        print(cpps)
        assert len(cpps) == 1
        assert anno02 in cpps[0]['chosen'][-1]["content"]
        assert anno01 in cpps[0]['rejected'][-1]["content"] or anno03 in cpps[0]['rejected'][-1]["content"]
        assert anno02 not in cpps[0]['rejected'][-1]["content"]


@pytest.mark.asyncio
class TestArgannoFailureTypePreferencePairGenerator:

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            (
                """
                ```xml
                <proposition id="1">We should stop eating meat.</proposition>
                                
                <proposition id="2">Animals suffer.</proposition> <proposition id="2" supports="1 2">Animal farming causes climate change.</proposition>
                ```
                """,
                """
                ```xml
                <proposition id="1">We should stop eating meat.</proposition>
                                
                <proposition id="2">Animals suffer.</proposition> <proposition id="2" supports="1 4">Animal farming causes climate change.</proposition>
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
    ):
        
        await HirpoTester.test_generic_failure_type_preference_generator(
            problem_class,
            solution_class,
            judge_class,
            source_texts,
            chosen,
            rejected,
        )        
