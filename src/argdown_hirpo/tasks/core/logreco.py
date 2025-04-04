from typing import Any, Sequence

import dataclasses
from textwrap import dedent
import textdistance

from pyargdown import (
    ArgdownMultiDiGraph,
    Conclusion,
    Proposition,
    parse_argdown,
)

from argdown_hirpo.base import (
    Problem,
    ScoringVirtuePreferencePairGenerator,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    FeedbackGenerator,
)

from argdown_hirpo.logic.logic import get_propositional_variables
from argdown_hirpo.verifiers.logreco_verifier import LogRecoVerifier, DEFAULT_EVAL_DIMENSIONS_MAP
from argdown_hirpo.logic.fol_to_nl import FOL2NLTranslator
    


class LogRecoProblem(Problem):
    """Task: Reconstruct the main argument as deductively valid using premise conclusion structure and including formalization."""

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
            Assignment: Reconstruct a source text's main line of reasoning as a deductively valid argument in standard form.
                        
            Logically reconstruct the main argument in the following source text. Formalize all the premises and conclusions.
            Make sure the reconstructed argument is deductively valid and all premises are relevant.

            ::: {{.source_text}}              
            {sources}
            :::

            Note in particular:

            - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
              ending with '```'. Just include a single Argdown codeblock in your answer.                                            

            - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).

            - For each proposition in your reconstruction (premises and conclusions), provide an adequate FOL formalization in NLTK
              syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Minimal example:
              `(1) Socrates is mortal. {{formalization: 'F(a)', declarations: {{'a': 'Socrates', 'F': 'being mortal'}} }}`.
              Only declare variables that are used in the corresponding formalization and that have not been declared before.
              Ensure that your formalizations are consistent with each other.

            - For each inference step in the argument, provide information about which previously introduced premises or 
              conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
            
            - You may, but are in no way required to add additional information about which inference rules or argumentation
              schemes are applied in each sub-argument.

            - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
              summarize its gist in line with Argdown syntax conventions. 

            - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument.
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
class LogicalReco(Solution):
    """Solution to the argument analysis problem: an argdown snippet."""

    argdown_snippet: str

    def __str__(self):
        return self.argdown_snippet
    
    @classmethod
    def from_raw_answer(cls, answer) -> "LogicalReco":
        """Extract a LogicalReco from a raw answer string."""
        if answer.count("```argdown") == 1:
            if answer.split("```argdown")[1].count("\n```") == 1:
                answer = answer.split("```argdown")[1].split("\n```")[0]
                answer = "```argdown" + answer + "\n```"
        return cls(argdown_snippet=answer)


class LogRecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return LogRecoProblem(inputs)
        raise ValueError(
            "Inputs to an argument reconstruction problem must be a string or a list of strings"
        )


class LogRecoJudge(Judge):
    """Judge for the informal argument reconstruction task."""

    def _evaluate_logreco(
        self, problem: LogRecoProblem, reco: LogicalReco
    ) -> Evaluation:
        is_valid = True
        artifacts: dict[str, Any] = {}
        eval_data = {
            "fenced_code_block": "",
            "invalid_argdown_syntax": "",
            "no_unique_argument": "",
            "illformed_argument": "",  # no pcs
            "missing_label_gist": "",
            "missing_inference_info": "",
            "unknown_proposition_references": "",  # in inference info
            "unused_propositions": "",  # unused propositions
            "disallowed_material": "", # more propositions
            "flawed_formalizations": "",  # missing, duplicate declarations etc. etc.
            "invalid_inference": "",  # invalid inference
            "redundant_premises": "",  # redundant premises
            "inconsistent_premises": "",  # inconsistent premises
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

        artifacts["argdown"] = argdown

        if argdown:

            verifier = LogRecoVerifier(argdown)
            artifacts["all_expressions"] = verifier.all_expressions
            artifacts["all_declarations"] = verifier.all_declarations
            check, msg = verifier.has_unique_argument()
            if check is False:
                eval_data["no_unique_argument"] = msg if msg else "No unique argument."

            del verifier

            logreco_evals, _, _ = LogRecoVerifier.run_battery(argdown)
            eval_data.update(logreco_evals)

            is_valid = not any(v for v in eval_data.values())

        return Evaluation(
            is_valid=is_valid, artifacts=artifacts, metrics=eval_data
        )

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(original_solution, LogicalReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, LogicalReco), (
                "All solutions must be LogicalReco objects"
            )
            evaluations.append(self._evaluate_logreco(problem, solution))

        return evaluations


class LogRecoFeedbackGenerator(FeedbackGenerator):
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
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(solution, LogicalReco), "Solution must be an LogicalReco"
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
        # remove empty and duplicate answers
        answers = [a for a in answers if a]
        answers = list(set(answers))

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class ManyIntermediateConclusionsPreferencePairGenerator(
    ScoringVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with more intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to include as many sub-arguments as possible. "
        "I.e., reconstruct the argument with many intermediate steps. That is what counts here."
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return number_intermediate_conclusions


class FewIntermediateConclusionsPreferencePairGenerator(
    ScoringVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with fewer intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to minimize the number of intermediate conclusions. "
        "I.e., reconstruct the argument with as few sub-arguments as possible. That is what counts here."
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return (number_intermediate_conclusions + 1) ** -1


class IndependentWordingPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with independent wording of arguments and claims."""

    hints = [
        "Make sure that you render the argument's premises and conclusion(s) *in your own words*, "
        "and independently from the formulations in the source text. This is crucial at this step."
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
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


class SourceTextProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the source text."""

    hints = [
        "Make sure that your argument reconstruction stays maximally faithful to and mimics closely the original source text!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(reco, LogicalReco), "Solution must be an LogicalReco"
        return round(
                textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, reco.argdown_snippet
            ),
            1
        )


class SimplicityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with succinct and simple propositions."""

    hints = [
        "Make sure that you keep each of the argument's premises and conclusion(s) simple and succinct. "
        "Short sentences are crucial at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) ** -1 if lengths else 0
    

class VerbosityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with elaborate and verbose propositions."""

    hints = [
        "Render the argument's premises and conclusion(s) in an elaborate and verbose way. "
        "Long sentences are strongly preferred at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) if lengths else 0



class FormalizationsFaithfulnessPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with formalizations that are similiar to the sentences being formalized."""

    hints = [
        "Reconstruct the argument in such a way that your logico-semantic analysis (formalizations and declarations) "
        "coheres with the actual wording of the premises and conclusion(s). In particular, formalize your argument's "
        "premises and conclusion(s) faithfully!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        all_expressions = evaluation.artifacts["all_expressions"]
        all_declarations = evaluation.artifacts["all_declarations"]


        dlds: list[float] = []
        for pr in argument.pcs:
            expression = next(
                expr for exprl, expr in all_expressions.items() if exprl == pr.label
            )
            proposition = next(
                p for p in argdown.propositions if p.label == pr.proposition_label
            )

            text_1 = FOL2NLTranslator.translate_to_nl_sentence(
                expression, all_declarations
            )

            for text_2 in proposition.texts:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_similarity(
                        text_1, text_2
                    )
                )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0


class PredicateLogicPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with formalizations that use but predicate logic."""

    hints = [
        "Formalize the premises and conclusions in your argument reconstruction "
        "using predicate logic. Avoid using propositional logic! No propositional variables!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        all_expressions = evaluation.artifacts["all_expressions"]
        if not all_expressions:
            return 0
        n_has_prop_vars = sum(bool(get_propositional_variables(expr)) for expr in all_expressions.values())
        return 1 - (n_has_prop_vars / len(all_expressions))



