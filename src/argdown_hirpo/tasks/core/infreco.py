from typing import Sequence

from abc import abstractmethod
import dataclasses
import random
from textwrap import dedent
import textdistance

from pyargdown import (
    ArgdownMultiDiGraph,
    Argument,
    Conclusion,
    DialecticalType,
    Proposition,
    parse_argdown,
)
from pyargdown.parser.base import ArgdownParser

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ChatPreferencePair,
    ProblemSolutionChat,
    ProblemGenerator,
    SolutionGenerator,
    Judge,
    FeedbackGenerator,
    VirtuePreferencePairGenerator,
)


class InfRecoProblem(Problem):
    """Task: Reconstruct the main argument as a premise conclusion structure, no formalization, no dialectics."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace and newlines
        sources = sources.strip("\n ")
        self.sources = sources

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            Assignment: Reconstruct a source text's main argument in standard form.
                        
            Identify the main argument in the following source text and reconstruct it as premise-conclusion structure using Argdown.

            ::: {{.source_text}}              
            {sources}
            :::

            Note in particular:

            - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
              ending with '```'. Just include a single Argdown codeblock in your answer.                                            
            - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
              conclusion, and possible intemediate conclusions).
            - For each conclusion in the argument, provide information about which previously introduced premises or 
              conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
            - You may, but are in no way required to add additional information about which inference rules or argumentation
              schemes are applied in each sub-argument.
            - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
              summarize its gist in line with Argdown syntax conventions. 
                   
            Carefully consider the following DON'Ts:

            - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument.
            - Do NOT add any inline dialectical relations in the premise conclusion structure.
            - Do NOT add any yaml inline data besides the required inference information.
            - Do NOT add any formalization of the argument's propositions (premises or conclusions) in your Argdown code.

        """)
            .strip()
            .format(sources=self.sources)
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
              "\n\n"
              "> [!WARNING]\n"
              "> For didactic purposes, I want you to make mistakes in your answer, violating the above instructions.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"


        return prompt

    def revise_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = "Revise your previously submitted argument reconstruction given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
              "\n\n"
              "> [!WARNING]\n"
              "> For didactic purposes, I still want you to make mistakes in your revised answer.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"

        return prompt


@dataclasses.dataclass
class InformalReco(Solution):
    """Solution to the argument analysis problem: an argdown snippet."""

    argdown_snippet: str

    def __str__(self):
        return self.argdown_snippet


class InfRecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return InfRecoProblem(inputs)
        raise ValueError(
            "Inputs to an argument recinstruction problem must be a string or a list of strings"
        )


class InfRecoSolutionGenerator(SolutionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_solutions = kwargs.get("n_solutions", 10)
        self.temperature = kwargs.get("temperature", 0.5)
        self.max_tokens = kwargs.get("max_tokens", 2048)

    async def arun(
        self,
        problem: InfRecoProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[InformalReco]:
        assert isinstance(original_solution, InformalReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for revised solutions"
        )

        messages = [
            {
                "role": "user",
                "content": problem.instruct_prompt(),
            }
        ]

        if original_solution and feedback:
            messages += [
                {
                    "role": "assistant",
                    "content": str(original_solution),
                },
                {
                    "role": "user",
                    "content": feedback.prompt,
                },
                {
                    "role": "assistant",
                    "content": feedback.feedback,
                },
                {
                    "role": "user",
                    "content": problem.revise_prompt(),
                },
            ]

        answers = await self._generate(
            messages,
            max_tokens=self.max_tokens,
            n=self.n_solutions,
            temperature=self.temperature,
        )

        recos: list[InformalReco] = []

        # postprocess: extract fenced code block
        for answer in answers:
            if answer.count("```argdown") == 1:
                if answer.split("```argdown")[1].count("\n```") == 1:
                    answer = answer.split("```argdown")[1].split("\n```")[0]
                    answer = "```argdown" + answer + "\n```"
            recos.append(InformalReco(argdown_snippet=answer))

        return recos


class InfRecoJudge(Judge):
    """Judge for the informal argument reconstruction task."""

    def _evaluate_infreco(
        self, problem: InfRecoProblem, reco: InformalReco
    ) -> Evaluation:
        is_valid = True
        eval_data = {
            "fenced_code_block": "",
            "invalid_argdown_syntax": "",
            "no_unique_argument": "",
            "illformed_argument": "",  # starts with conclusion / ends with premise
            "missing_label_gist": "",
            "missing_inference_info": "",
            "unknown_proposition_references": "",  # in inference info
            "disallowed_material": "", # more propositions, inline dialectical relations, yaml inline data
        }

        ads = reco.argdown_snippet.strip("\n ")
        if ads.startswith("```argdown") and ads.endswith("```"):
            ads = "\n".join(ads.splitlines()[1:-1])
        else:  # no fenced code block
            is_valid = False
            error_msg = "Failed to extract single fenced argdown block:"
            if ads.count("```argdown") == 0:
                error_msg += " No fenced code block starting with '```argdown'."
            if ads.count("```argdown") > 1:
                error_msg += (
                    " More than one fenced code block starting with '```argdown'."
                )
            if "```\n" not in ads:
                error_msg += " No closing '```'."
            eval_data["fenced_code_block"] = error_msg

        try:
            argdown = parse_argdown(ads)
        except Exception as e:
            argdown = None
            is_valid = False
            eval_data["invalid_argdown_syntax"] = f"Failed to parse argdown: {str(e)}"

        if argdown:
            argument: Argument | None = None
            if len(argdown.arguments) == 0:
                is_valid = False
                eval_data["no_unique_argument"] = "No argument in the argdown snippet."
            elif len(argdown.arguments) > 1:
                is_valid = False
                eval_data["no_unique_argument"] = "More than one argument in argdown snippet."
            else:
                argument = argdown.arguments[0]

            if argument:
                if not argument.pcs:
                    is_valid = False
                    eval_data["illformed_argument"] = (
                        "Argument lacks premise conclusion structure, i.e., is not reconstructed in standard form."
                    )

                if argument.pcs:
                    msg = []
                    if isinstance(argument.pcs[0], Conclusion):
                        msg.append("Argument starts with a conclusion, not a premise.")
                    if not isinstance(argument.pcs[-1], Conclusion):
                        msg.append("Argument does not end with a conclusion.")
                    if len(argument.gists) > 1:
                        msg.append("More than one gist for the argument.")
                    pcs_labels = [p.label for p in argument.pcs]
                    for label in pcs_labels:
                        if pcs_labels.count(label) > 1:
                            msg.append(f"Duplicate label '{label}' in the argument's standard form.")
                    if msg:
                        is_valid = False
                        eval_data["illformed_argument"] = " ".join(msg)
                    del msg


                msg = []
                if ArgdownParser.is_unlabeled(argument):
                    msg.append("Argument lacks a label / title.")
                if not argument.gists:
                    msg.append("Argument lacks a gist / summary.")
                if msg:
                    is_valid = False
                    eval_data["missing_label_gist"] = " ".join(msg)
                del msg

            if argument and argument.pcs:
                msg = []
                for c in argument.pcs:
                    if isinstance(c, Conclusion):
                        inf_data = c.inference_data
                        if not inf_data:
                            msg.append(f"Conclusion {c.label} lacks yaml inference information.")
                        else:
                            from_list = inf_data.get("from")
                            if from_list is None:
                                msg.append(f"Conclusion {c.label} inference information lacks 'from' key.")
                            elif not isinstance(from_list, list):
                                msg.append(f"Conclusion {c.label} inference information 'from' value is not a list.")
                            elif len(from_list) == 0:
                                msg.append(f"Conclusion {c.label} inference information 'from' value is empty.")
                if msg:
                    is_valid = False
                    eval_data["missing_inference_info"] = " ".join(msg)
                del msg

            if argument and argument.pcs:
                msg = []
                for enum, c in enumerate(argument.pcs):
                    if isinstance(c, Conclusion):
                        inf_data = c.inference_data
                        from_list = inf_data.get("from", [])
                        if isinstance(from_list, list):
                            for ref in from_list:
                                if str(ref) not in [p.label for p in argument.pcs[:enum]]:
                                    msg.append(
                                        f"Item '{ref}' in inference information of conclusion {c.label} does "
                                        "not refer to a previously introduced premise or conclusion."
                                    )
                if msg:
                    is_valid = False
                    eval_data["unknown_proposition_references"] = " ".join(msg)
                del msg

            msg = []
            if argument and argument.pcs:
                if len(argdown.propositions) > len(argument.pcs):
                    msg.append(
                        "Argdown snippet contains propositions other than the ones in the argument."
                    )
            else: 
                if len(argdown.propositions) > 0:
                    msg.append(
                        "Argdown snippet contains propositions outside the argument."
                    )
            if any(
                set(d.dialectics) != {DialecticalType.GROUNDED}
                for d in argdown.dialectical_relations
            ):
                msg.append(
                    "Argdown snippet defines dialectical relations."
                )
            if any(prop.data for prop in argdown.propositions):
                msg.append("Some propositions contain yaml inline data.")
            if any(arg.data for arg in argdown.arguments):
                msg.append("Some arguments contain yaml inline data.")
            if msg:
                is_valid = False
                eval_data["disallowed_material"] = " ".join(msg)
            del msg

        return Evaluation(
            is_valid=is_valid, artifacts={"argdown": argdown}, metrics=eval_data
        )

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(original_solution, InformalReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, InformalReco), (
                "All solutions must be InformalReco objects"
            )
            evaluations.append(self._evaluate_infreco(problem, solution))

        return evaluations


class InfRecoFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_solutions", 5)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given argument reconstruction.

            You will be shown an argument analysis problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

                                                
            ## Problem Statement
            {problem}

            
            ## Student's Solution
            {solution}

            
            ## Evaluation
            The student's solution is NOT valid.
            Particular issues:
            {evaluation_issues}

            
            Given this information, provide feedback to the student and clear instructions for how to improve the solution.
        """).format(
            problem=problem.instruct_prompt(),
            solution=str(solution),
            evaluation_issues=evaluation_issues,
        )

        answers = await self._generate(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_tokens,
            n=self.n_feedbacks,
            temperature=self.temperature,
        )

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class InfRecoVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the informal argument reconstruction task."""

    hints: list[str] = []

    @abstractmethod
    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        pass

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert all(isinstance(s, InformalReco) for s in candidate_solutions), (
            "All solutions must be InformalReco objects"
        )
        assert original_solution is None or isinstance(
            original_solution, InformalReco
        ), "Original solution must be an InformalReco"
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )

        pairs: list[ChatPreferencePair] = []

        # rank valid argmaps according to the _score function
        valid_recos: list[tuple[InformalReco, Evaluation]] = list(
            zip(candidate_solutions, evaluations)  # type: ignore
        )
        valid_recos.sort(key=lambda x: self._score(problem, x[0], x[1]), reverse=True)
        valid_recos = [
            (solution, evaluation)
            for solution, evaluation in valid_recos
            if evaluation.is_valid
        ]

        if len(valid_recos) < 2:
            return pairs
        top_score = self._score(problem, *valid_recos[0])
        if top_score == self._score(problem, *valid_recos[-1]):
            return pairs

        top_reco, _ = valid_recos[0]
        weaker_reco = random.choice(
            [s for s, e in valid_recos if self._score(problem, s, e) < top_score]
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=weaker_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
            )
        )

        return pairs



class NoUnusedPropsPreferencePairGenerator(InfRecoVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with fewer unused premises or conclusions."""

    hints = [
        "In your argument reconstruction, make sure that every premise and every intermediate conclusion is "
        "(explicitly) used in a subsequent inference. (Every unused premise or conclusion counts as a mistake.)"
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        used_labels = set()
        for c in argument.pcs:
            if isinstance(c, Conclusion):
                used_labels.update(c.inference_data.get("from", []))
        number_unused_props = sum(1 for p in argument.pcs[:-1] if p.label not in used_labels)

        return (number_unused_props + 1) ** -1


class ManyIntermediateConclusionsPreferencePairGenerator(
    InfRecoVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with more intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to include as many sub-arguments as possible. "
        "I.e., reconstruct the argument with many intermediate steps. That is what counts here."
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return number_intermediate_conclusions


class FewIntermediateConclusionsPreferencePairGenerator(
    InfRecoVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with fewer intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to minimize the number of intermediate conclusions. "
        "I.e., reconstruct the argument with as few sub-arguments as possible. That is what counts here."
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return (number_intermediate_conclusions + 1) ** -1


class IndependentWordingPreferencePairGenerator(InfRecoVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with independent wording of arguments and claims."""

    hints = [
        "Make sure that you render the argument's premises and conclusion(s) *in your own words*, "
        "and independently from the formulations in the source text. This is crucial at this step."
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        dlds: list[float] = []
        for p in propositions:
            for t in p.texts:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_distance(
                        problem.sources, t
                    )
                )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0


class SourceTextProximityPreferencePairGenerator(InfRecoVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the source text."""

    hints = [
        "Make sure that your argument reconstruction stays maximally faithful to and mimics closely the original source text!"
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        return round(
                textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, reco.argdown_snippet
            ),
            1
        )


class SimplicityPreferencePairGenerator(InfRecoVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with succinct and simple propositions."""

    hints = [
        "Make sure that you keep each of the argument's premises and conclusion(s) simple and succinct. "
        "Short sentences are crucial at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) ** -1 if lengths else 0
    

class VerbosityPreferencePairGenerator(InfRecoVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with elaborate and verbose propositions."""

    hints = [
        "Render the argument's premises and conclusion(s) in an elaborate and verbose way. "
        "Long sentences are strongly preferred at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: InfRecoProblem,
        reco: InformalReco,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) if lengths else 0


