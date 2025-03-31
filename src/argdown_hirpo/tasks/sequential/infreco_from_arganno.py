from textwrap import dedent
import textdistance

from argdown_hirpo.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_hirpo.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco
)
from argdown_hirpo.tasks.core.arganno import (
    AnnotationProblemGenerator,
    AnnotationSolutionGenerator,
    AnnotationJudge,
)


class InfRecoFromArgAnnoProblem(InfRecoProblem):
    """
    Task: Reconstruct the main argument as a premise conclusion structure,
    no formalization, no dialectics.
    Input: Source text and argumentative annotation.
    """

    def __init__(self, annotation: str):
        if isinstance(annotation, list):
            annotation = "\n\n-----\n\n".join(annotation)
        # remove leading and trailing whitespace and newlines
        annotation = annotation.strip("\n ")
        self.annotation = annotation
        self.sources = annotation

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            Assignment: Reconstruct a source text's main argument in standard form.
                        
            Use the argumentative annotation to identify the main argument in the following source text and reconstruct it as premise-conclusion structure using Argdown.

            ::: {{.source_text}}              
            {sources}
            :::

            Note in particular:

            - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
              ending with '```'. Just include a single Argdown codeblock in your answer.                                            
            - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
              conclusion, and possible intemediate conclusions), no matter whether the annotation highlights more than
              one argument.
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
            .format(sources=self.annotation)
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


class InfRecoFromArgAnnoProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arganno_pg = AnnotationProblemGenerator()
        self._arganno_sg = AnnotationSolutionGenerator(*args, **kwargs)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            arganno_problem = await self._arganno_pg.arun(inputs)
            arganno_solution = await self._arganno_sg.arun(arganno_problem)
            return InfRecoFromArgAnnoProblem(str(arganno_solution))
        raise ValueError(
            "Inputs to an argument reconstruction problem must be a string or a list of strings"
        )


class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the source text's annotation."""

    hints = [
        "Make sure that your argument reconstruction stays faithful to and mimics closely "
        "the annotation of the source text. In particular, try to use supporting propositions from the annotation "
        "as premises / intermediate conclusions in your argument reconstruction!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, InfRecoFromArgAnnoProblem)
        assert isinstance(reco, InformalReco)

        soup, _ = AnnotationJudge().parse_xml_snippet(problem.annotation)
        anno_props = soup.find_all("proposition")
        supporting_props = [
            ap
            for ap in anno_props
            if ap.get("supports")  # type: ignore
        ]
        list_anno_props = "\n".join([ap.text for ap in supporting_props])

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                list_anno_props, reco.argdown_snippet
            ),
            1,
        )
