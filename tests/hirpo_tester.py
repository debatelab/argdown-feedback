from pprint import pprint
import textwrap
import networkx as nx  # type: ignore


from argdown_feedback.tasks.base import Evaluation, Feedback, GenericFailureDiffPreferencePairGenerator, Problem, Solution, HIRAbstractGeneratorLLM


class HirpoTester:
    @staticmethod
    async def test_problem_generator(
        problem_generator_class,
        problem_class,
        source_texts,
        model_kwargs=None,
        keeps_source_texts=True,
    ):
        pg = (
            problem_generator_class()
            if model_kwargs is None
            else problem_generator_class(**model_kwargs)
        )
        problem = await pg.arun(source_texts)
        assert isinstance(problem, problem_class)

        print(problem.instruct_prompt())
        print(problem.revise_prompt())

        if keeps_source_texts:
            assert source_texts[0] in problem.instruct_prompt()
            assert source_texts[0] in problem.instruct_prompt(ask_for_invalid=True)
        assert "super cool hint" in problem.instruct_prompt(hints=["super cool hint"])

        assert "!WARNING" in problem.instruct_prompt(ask_for_invalid=True)
        assert "!WARNING" in problem.revise_prompt(ask_for_invalid=True)

        inv_prompt = problem.instruct_prompt(
            ask_for_invalid=True,
            evaluation=Evaluation(
                is_valid=False, artifacts={}, metrics={"level": "AAA"}
            ),
        )
        assert "!WARNING" in inv_prompt
        assert "level" in inv_prompt
        assert "AAA" in inv_prompt

        inv_prompt = problem.revise_prompt(
            ask_for_invalid=True,
            evaluation=Evaluation(
                is_valid=False, artifacts={}, metrics={"level": "AAA"}
            ),
        )
        assert "!WARNING" in inv_prompt
        assert "level" in inv_prompt
        assert "AAA" in inv_prompt

    @staticmethod
    async def test_solution_generator(
        problem_generator_class,
        solution_generator_class,
        solution_class,
        source_texts,
        model_kwargs,
    ):
        if issubclass(problem_generator_class, HIRAbstractGeneratorLLM):
            pg = problem_generator_class(**model_kwargs)
        else:
            pg = problem_generator_class()
        sg = solution_generator_class(
            solution_class=solution_class, n_solutions=1, **model_kwargs
        )  # lmstudio server does not support param n
        problem = await pg.arun(source_texts)
        solutions = await sg.arun(problem)
        assert len(solutions) == 1
        for i, sol in enumerate(solutions):
            print(f"## Solution {i + 1}")
            print(sol)
            assert isinstance(sol, solution_class)

    @staticmethod
    async def test_judge_valid(
        problem_generator_class, judge_class, valid_recos, source_texts, argdown_artifact_keys=["argdown"], model_kwargs = None
    ):
        source_text = source_texts[0] if isinstance(source_texts, list) else source_texts
        if issubclass(problem_generator_class, HIRAbstractGeneratorLLM):
            assert model_kwargs is not None, "model_kwargs must be provided for HIRAbstractGeneratorLLM"
            pg = problem_generator_class(**model_kwargs)
        else:
            pg = problem_generator_class()
        problem = await pg.arun(source_text)
        judge = judge_class()
        evaluations = await judge.arun(problem, valid_recos)
        assert len(evaluations) == len(valid_recos)
        for i, ev in enumerate(evaluations):
            print(f"## Solution {i + 1}")
            pprint(ev)
            for ad_key in argdown_artifact_keys:
                if ad_key in ev.artifacts:
                    print(f"## {ad_key}")
                    argdown = ev.artifacts.get(ad_key)
                    #pprint(argdown.dialectical_relations)
                    #print(nx.node_link_data(argdown))
                    assert argdown
            assert ev.is_valid
            assert not any(v for _, v in ev.metrics.items())

    @staticmethod
    async def test_judge_valid2(
        example_problem,
        judge_class,
        valid_recos,
        argdown_artifact_keys=["argdown"],
        model_kwargs = None
    ):
        problem = example_problem
        judge = judge_class()
        evaluations = await judge.arun(problem, valid_recos)
        assert len(evaluations) == len(valid_recos)
        for i, ev in enumerate(evaluations):
            print(f"## Solution {i + 1}")
            pprint(ev)
            for ad_key in argdown_artifact_keys:
                if ad_key in ev.artifacts:
                    #print(f"## {ad_key}")
                    argdown = ev.artifacts.get(ad_key)
                    #pprint(argdown.dialectical_relations)
                    #print(nx.node_link_data(argdown))
                    assert argdown
            assert ev.is_valid
            assert not any(v for _, v in ev.metrics.items())
        


    @staticmethod
    async def test_judge_invalid(
        problem_generator_class, judge_class, invalid_recos, source_texts, model_kwargs = None
    ):
        source_text = source_texts[0]
        if issubclass(problem_generator_class, HIRAbstractGeneratorLLM):
            assert model_kwargs is not None, "model_kwargs must be provided for HIRAbstractGeneratorLLM"
            pg = problem_generator_class(**model_kwargs)
        else:
            pg = problem_generator_class()
        problem = await pg.arun(source_text)
        judge = judge_class()
        evaluations = await judge.arun(problem, invalid_recos)
        assert len(evaluations) == len(invalid_recos)
        for i, ev in enumerate(evaluations):
            print(f"## Solution {i + 1}")
            pprint(ev)
            argdown = ev.artifacts.get("argdown")
            if argdown:
                print(argdown.propositions)
                print(argdown.arguments)
            assert not ev.is_valid
            assert any(v for _, v in ev.metrics.items())

    @staticmethod
    async def test_judge_invalid2(
        example_problem,
        judge_class,
        invalid_recos,
        argdown_artifact_keys=["argdown"],
        model_kwargs = None
    ):
        problem = example_problem
        judge = judge_class()
        evaluations = await judge.arun(problem, invalid_recos)
        assert len(evaluations) == len(invalid_recos)
        for i, ev in enumerate(evaluations):
            print(f"## Solution {i + 1}")
            pprint(ev)
            for ad_key in argdown_artifact_keys:
                if ad_key in ev.artifacts:
                    #print(f"## {ad_key}")
                    argdown = ev.artifacts.get(ad_key)
                    #pprint(argdown.dialectical_relations)
                    print(nx.node_link_data(argdown) if argdown else "argdown=None")
            assert not ev.is_valid
            assert any(v for _, v in ev.metrics.items())

    @staticmethod
    async def test_feedback_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        invalid_recos,
        source_texts,
        model_kwargs,
    ):
        source_text = source_texts[0]
        if issubclass(problem_generator_class, HIRAbstractGeneratorLLM):
            assert model_kwargs is not None, "model_kwargs must be provided for HIRAbstractGeneratorLLM"
            pg = problem_generator_class(**model_kwargs)
        else:
            pg = problem_generator_class()
        problem = await pg.arun(source_text)

        judge = judge_class()
        evaluations = await judge.arun(problem, invalid_recos)

        fg = feedback_generator_class(n_feedbacks=1, **model_kwargs)
        for reco, evaluation in zip(invalid_recos, evaluations):
            feedbacks = await fg.arun(problem, reco, evaluation)
            assert len(feedbacks) == 1
            feedback = feedbacks[0]
            assert isinstance(feedback, Feedback)
            assert problem.instruct_prompt() in feedback.prompt
            assert str(reco) in feedback.prompt
            print(feedback)

    @staticmethod
    async def test_revised_solution_generator(
        problem_generator_class,
        judge_class,
        feedback_generator_class,
        solution_generator_class,
        solution_class,
        invalid_recos,
        source_texts,
        model_kwargs,
    ):
        source_text = source_texts[0]
        if issubclass(problem_generator_class, HIRAbstractGeneratorLLM):
            assert model_kwargs is not None, "model_kwargs must be provided for HIRAbstractGeneratorLLM"
            pg = problem_generator_class(**model_kwargs)
        else:
            pg = problem_generator_class()
        problem = await pg.arun(source_text)
        original_solution = invalid_recos[-1]

        judge = judge_class()
        evaluations = await judge.arun(problem, [original_solution])
        evaluation = evaluations[0]

        fg = feedback_generator_class(n_feedbacks=1, **model_kwargs)
        feedbacks = await fg.arun(problem, original_solution, evaluation)
        feedback = feedbacks[0]

        sg = solution_generator_class(
            solution_class=solution_class, n_solutions=1, **model_kwargs
        )  # lmstudio server does not support param n
        revised_solutions = await sg.arun(
            problem=problem, original_solution=original_solution, feedback=feedback
        )
        assert len(revised_solutions) == 1
        revised_solution = revised_solutions[0]
        print(revised_solution)
        assert isinstance(revised_solution, solution_class)

    @staticmethod
    async def test_generic_failure_type_preference_generator(
        problem_class,
        solution_class,
        judge_class,
        source_texts,
        chosen,
        rejected,
        example_problem = None,
    ):
        if example_problem is not None:
            problem = example_problem
        else:
            problem = problem_class(source_texts[0])

        judge = judge_class()
        ppg = GenericFailureDiffPreferencePairGenerator()


        if not isinstance(chosen, solution_class):
            snippet_chosen = textwrap.dedent(chosen)
            chosen = solution_class(snippet_chosen)

        if not isinstance(rejected, solution_class):
            snippet_rejected = textwrap.dedent(rejected)
            rejected = solution_class(snippet_rejected)

        candidate_solutions = [chosen, rejected]

        pprint(problem)
        pprint(candidate_solutions)
        pprint(judge)
        evaluations = await judge.arun(problem, candidate_solutions)
        pprint(evaluations)
        
        cpps = await ppg.arun(problem, candidate_solutions, evaluations)
        pprint(cpps)
        assert len(cpps) == 1
        assert ppg.avoid_errors_hint in cpps[0]["chosen"][0]["content"]
        assert ppg.avoid_errors_hint in cpps[0]["rejected"][0]["content"]
        assert str(chosen) in cpps[0]["chosen"][-1]["content"]
        assert str(rejected) in cpps[0]["rejected"][-1]["content"]
