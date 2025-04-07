from textwrap import dedent
from typing import Any
from bs4 import BeautifulSoup
from pyargdown import ArgdownMultiDiGraph, Argument, Proposition, Valence
import textdistance

from argdown_hirpo.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    GenericSolutionGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_hirpo.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap
)
from argdown_hirpo.tasks.core.arganno import (
    Annotation,
    AnnotationProblemGenerator,
    AnnotationJudge,
)


class ArgmapFromArgannoProblem(ArgMapProblem):
    """
    Task: Reconstruct the main argument as an informal argument map.
    Input: Argumentative text annotation.
    """

    def __init__(self, annotated_text: str, soup_anno: BeautifulSoup | None = None):
        annotated_text = annotated_text.strip("\n ")
        self.annotated_text = annotated_text
        self.sources = annotated_text
        self.soup_anno = soup_anno

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            Assignment: Reconstruct a source text's argumentation as a argument map.
                        
            Use the following argumentative annotation to reconstruct the text's arguments as an informal Argdown argument map.

            ::: {{.source_text}}              
            {sources}
            :::

            In particular, I ask you

            - to explicitly label all nodes in your argument map;
            - to use square/angled brackets for labels to distinguish arguments/claims;
            - to indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
            - to refer, in your argument, to text segments in the annotation through yaml inline data with an `annotation_ids` attribute that contains a list of proposition `ids`.
            - NOT to include any detailed reconstructions of individual arguments as premise-conclusion-structures in your argdown code.

            Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.                                                
        """)
            .strip()
            .format(sources=self.annotated_text)
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


class ArgmapFromArgannoProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arganno_pg = AnnotationProblemGenerator()        
        self._arganno_sg = GenericSolutionGenerator(solution_class=Annotation, *args, **kwargs, n_solutions=1)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            arganno_problem = await self._arganno_pg.arun(inputs)
            arganno_solution = await self._arganno_sg.arun(arganno_problem)
            soup_anno, _ = AnnotationJudge().parse_xml_snippet(arganno_solution[0].annotated_source_text)
            return ArgmapFromArgannoProblem(
                annotated_text=str(arganno_solution),
                soup_anno=soup_anno,
            )
        raise ValueError(
            "Inputs to an argument reconstruction problem must be a string or a list of strings"
        )


class AnnotationTextProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument mapping task, prefering valid argument maps
    that stick closely to the source text's annotation."""

    hints = [
        "Make sure that your argument map stays faithful to and mimics closely "
        "the annotation of the source text in terms of text flow and wording. "
        "In particular, try to use annotated propositions segments "
        "as gists for arguments and claims in your map!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgmapFromArgannoProblem)
        assert isinstance(reco, ArgumentMap)

        soup = problem.soup_anno
        if soup is None:
            return 0
        anno_props = soup.find_all("proposition")
        list_anno_props = "\n".join([ap.text for ap in anno_props])

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                list_anno_props, reco.argdown_snippet
            ),
            1,
        )



class AnnotationGraphProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument mapping task, prefering valid argument maps
    that are structurally similar to the source text's annotation."""

    hints = [
        "Make sure that your argument map stays faithful to and mimics closely "
        "the annotation of the source text in terms of overall argumentative structure. "
        "In particular, match argument and propositions via `annotation_ids` references, and "
        "reproduce the annotated dialectic relations in your map!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgmapFromArgannoProblem)
        assert isinstance(reco, ArgumentMap)

        soup = problem.soup_anno        
        argdown: ArgdownMultiDiGraph | None = evaluation.artifacts.get("argdown_map")
        if soup is None or argdown is None:
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
        def anno_ids(prop_label: str) -> list:
            node: Proposition | Argument | None
            node = argdown.get_proposition(prop_label)
            if node is None:
                node = argdown.get_argument(prop_label)
            if node is None:
                return []            
            return node.data.get("annotation_ids", [])

        matched_n = 0
        for from_id, to_id in supports_anno:
            if any(
                drel.valence == Valence.SUPPORT
                and from_id in anno_ids(drel.source)
                and to_id in anno_ids(drel.target)
                for drel in argdown.dialectical_relations                
            ):
                matched_n += 1

        for from_id, to_id in attacks_anno:
            if any(
                drel.valence == Valence.ATTACK
                and from_id in anno_ids(drel.source)
                and to_id in anno_ids(drel.target)
                for drel in argdown.dialectical_relations                
            ):
                matched_n += 1

        print("MATCHED_N", matched_n)

        return round(
            matched_n / (len(supports_anno) + len(attacks_anno)),
            1,
        )

