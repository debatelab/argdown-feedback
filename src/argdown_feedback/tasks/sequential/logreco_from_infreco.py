from textwrap import dedent
from pyargdown import Argdown
import textdistance

from argdown_feedback.tasks.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    GenericSolutionGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.infreco import (
    InfRecoProblemGenerator,
    InfRecoJudge,
    InformalReco,
)
from argdown_feedback.tasks.core.logreco import (
    LogicalReco,
    LogRecoProblem,
)


class LogrecoFromInfrecoProblem(LogRecoProblem):
    """
    Task: Logically analyse, revise and formalize the informally reconstructed argument.
    Input: infreco.
    """

    def __init__(
        self,
        argdown_snippet: str,
        argdown_infreco: Argdown | None = None,
        infreco_evaluation: Evaluation | None = None,
    ):
        argdown_snippet = argdown_snippet.strip()
        self.argdown_snippet = argdown_snippet
        self.argdown_infreco = argdown_infreco
        self.infreco_evaluation = infreco_evaluation
        self.sources = argdown_snippet


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            Assignment: Logically analyse an informally reconstructed argument.
                        
            Your task is to logically analyze and, as necessary, revise, the following informally reconstructed argument using Argdown.
            Formalize all the premises and conclusions in your revision. Make sure the revised argument is deductively valid and all 
            premises are relevant.

            ::: {{.informal_argument}}              
            {argdown_snippet}
            :::

            Note in particular:

            - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
              ending with '```'. Just include a single Argdown codeblock in your answer.

            - In your Argdown snippet, only include *a single argument* which re-analyzes the informal reconstruction (adding/removing or revising 
              premises, final conclusion, and possible intermediate conclusions as required).

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
        """)
            .strip()
            .format(argdown_snippet=self.argdown_snippet)
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

        return prompt


class LogrecoFromInfrecoProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._infreco_pg = InfRecoProblemGenerator()        
        self._infreco_sg = GenericSolutionGenerator(solution_class=InformalReco, *args, **kwargs, n_solutions=1)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            infreco_problem = await self._infreco_pg.arun(inputs)
            infreco_solution = await self._infreco_sg.arun(infreco_problem)            
            infreco_evaluation = InfRecoJudge()._evaluate_solution(infreco_solution[0], infreco_problem)
            argdown_infreco = infreco_evaluation.artifacts.get("argdown_reco")
            return LogrecoFromInfrecoProblem(
                argdown_snippet=str(infreco_solution[0]),
                argdown_infreco=argdown_infreco,
                infreco_evaluation=infreco_evaluation,
            )
        raise ValueError(
            "Inputs to an LogrecoFromInfrecoProblem must be a string or a list of strings"
        )


class InfrecoProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the original informal reconstruction."""

    hints = [
        "Make sure that your logical analysis stays faithful to and mimics closely "
        "the original informal reconstruction. In particular, try to re-use premises and conclusions "
        "where possible, or apply minimal changes to the informal reconstruction.",
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, LogrecoFromInfrecoProblem)
        assert isinstance(reco, LogicalReco)

        argdown_infreco = problem.argdown_snippet
        argdown_logreco = reco.argdown_snippet

        if not argdown_infreco or not argdown_logreco:
            return 0.0

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                argdown_infreco, argdown_logreco
            ),
            1,
        )
