from textwrap import dedent
from typing import Any
from bs4 import BeautifulSoup
from pyargdown import ArgdownMultiDiGraph, Valence
import textdistance

from argdown_feedback.tasks.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    GenericSolutionGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.arganno import (
    Annotation,
    AnnotationProblem,
    ANNOTATION_SCHEME,
)
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblemGenerator,
    ArgMapJudge,
    ArgumentMap
)


class ArgannoFromArgmapProblem(AnnotationProblem):
    """
    Task: Annotate a source text given an informal argument map.
    Input: Source text and argument map.
    """

    def __init__(
        self,
        sources: str | list[str],
        argdown_snippet: str,
        argdown_map: ArgdownMultiDiGraph | None = None,
        argmap_evaluation: Evaluation | None = None,
        strip_html: bool = True,
    ):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        if strip_html:
            sources = BeautifulSoup(sources, "html.parser").get_text()
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources
        self.argdown_snippet = argdown_snippet
        self.argdown_map = argdown_map
        self.argmap_evaluation = argmap_evaluation

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        qualifier = "(arguably imperfect) " if self.argmap_evaluation and not self.argmap_evaluation.is_valid else ""
        prompt = (
            dedent("""
            Assignment: Apply a given annotation scheme to a source text.
                        
            Annotate the following **source text** in order to identify the argumentative function of different parts in the text.

            ::: {{.source_text}}              
            {sources}
            :::

            This {qualifier}argument map sketches the source text's argumentative structure:

            {argdown_snippet}
                                      
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            In particular:           
        
            1. Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).                        
            2. Use the `argument_label` attribute to relate the annotated text segments to the given informal argument map.
            3. Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'.
            """)
        ).strip().format(
            sources=self.sources,
            argdown_snippet=self.argdown_snippet,
            annotation_scheme=ANNOTATION_SCHEME,
            qualifier=qualifier,
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

        return prompt


class ArgannoFromArgmapProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._argmap_pg = ArgMapProblemGenerator()        
        self._argmap_sg = GenericSolutionGenerator(solution_class=ArgumentMap, *args, **kwargs, n_solutions=1)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            argmap_problem = await self._argmap_pg.arun(inputs)
            argmap_solution = await self._argmap_sg.arun(argmap_problem)
            #print("argmap_problem", argmap_problem)
            #print("argmap_solution", argmap_solution)
            argmap_evaluation = ArgMapJudge()._evaluate_solution(argmap_solution[0], argmap_problem)
            argdown_map = argmap_evaluation.artifacts.get("argdown_map")
            return ArgannoFromArgmapProblem(
                sources=inputs,
                argdown_snippet=str(argmap_solution[0]),
                argdown_map=argdown_map,
                argmap_evaluation=argmap_evaluation,
            )
        raise ValueError(
            "Inputs to an ArgannoFromArgmapProblem must be a string or a list of strings"
        )


class ArgmapTextProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    that succeed in sticking closely to the informal argument map."""

    hints = [
        "Make sure that your annotation of the source text mimics closely "
        "the informal argument map in terms of text flow and wording. "
        "In particular, try to annotate propositions segments "
        "that, taken together, match the argument map!"
    ]

    def _score(
        self,
        problem: Problem,
        anno: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgannoFromArgmapProblem)
        assert isinstance(anno, Annotation)

        soup = evaluation.artifacts.get("soup")
        if soup is None:
            return 0
        anno_props = soup.find_all("proposition")
        list_anno_props = "\n".join([ap.text for ap in anno_props])

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                list_anno_props, problem.argdown_snippet
            ),
            1,
        )



class ArgmapGraphProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    that are structurally similar to the informal argument map."""

    hints = [
        "Make sure that your annotation of the source text stays faithful to and mimics closely "
        "the informal argument map in terms of overall argumentative structure. "
        "In particular, match arguments and claims via `argument_label` references, and "
        "reproduce the sketched dialectic relations in your annotation!"
    ]

    def _score(
        self,
        problem: Problem,
        anno: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgannoFromArgmapProblem)
        assert isinstance(anno, Annotation)

        argdown_map = problem.argdown_map
        soup = evaluation.artifacts.get("soup")
        if argdown_map is None or soup is None:
            return 0
        anno_props = soup.find_all("proposition")

        supports_anno: list[tuple[Any, Any]] = []
        attacks_anno: list[tuple[Any, Any]] = []
        for ap in anno_props:
            from_id = ap.get("id")  # type: ignore
            if from_id is None:
                continue
            for to_id in ap.get("supports", []):  # type: ignore
                if to_id is None:
                    continue
                supports_anno.append((ap["id"], to_id))  # type: ignore
            for to_id in ap.get("attacks", []):  # type: ignore
                if to_id is None:
                    continue
                attacks_anno.append((ap["id"], to_id))  # type: ignore

        # helper fn
        
        def arg_label(anno_id: str) -> str | None:
            if not anno_id:
                return None
            ap = next((a for a in anno_props if a.get("id") == anno_id), None)  # type: ignore
            if not ap:
                return None
            return ap.get("argument_label") # type: ignore


        matched_n = 0
        for drel in argdown_map.dialectical_relations:
            if any(
                drel.valence == Valence.SUPPORT
                and drel.source == arg_label(from_id)
                and drel.target == arg_label(to_id)
                for from_id, to_id in supports_anno                
            ):
                matched_n += 1

        for drel in argdown_map.dialectical_relations:
            if any(
                drel.valence == Valence.ATTACK
                and drel.source == arg_label(from_id)
                and drel.target == arg_label(to_id)
                for from_id, to_id in attacks_anno                
            ):
                matched_n += 1

        #print("MATCHED_N", matched_n)

        return round(
            matched_n / len(argdown_map.dialectical_relations),
            1,
        )

