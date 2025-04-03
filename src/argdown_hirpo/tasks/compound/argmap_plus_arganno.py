from typing import Any, Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import (
    ArgdownMultiDiGraph,
    Valence,
)
import textdistance

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_hirpo.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    Annotation,
    AnnotationJudge,
    AnnotationProblem,
)
from argdown_hirpo.tasks.core.argmap import (
    ArgMapJudge,
    ArgMapProblem,
    ArgumentMap,
)


class ArgmapPlusArgannoProblem(ArgMapProblem, AnnotationProblem):
    """Task: Create coherent argmap and arg annotation."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        sources = BeautifulSoup(sources, "html.parser").get_text()
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
            # Assignment: Annotate a source text and reconstruct its argumentation as an informal Argdown argument map.
                        
            Analyse the argumentation in the following **source text**. Create a coherent argumentative text annotation and a corresponding Argdown argument map.

            ::: {{.source_text}}              
            {sources}
            :::

            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way.
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                   
            ## Argument Mapping Task Details                   

            Create a syntactically correct informal Argdown argument map that reconstructs the overall argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

            Importantly, enclose your Argdown argument map in a fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation and Argument Map                                                

            The argument map and the annotated source text must cohere with each other. There should be a one-to-many correspondence between argument map nodes and annotated text segments. Moreover, the support and attack relations in the argument map should reflect the annotated dialectical relations.
                   
            In particular, you should ensure that: 

            - Every <proposition> element in the annotation has an `argument_label` attribute that refers to a node (label of claim or argument) in the argument map.
            - Every node in the Argdown argument map has yaml inline data with an `annotation_ids` attribute that contains a list of `id` attributes of the corresponding <proposition> element in the annotation.
            - Two nodes in the argument map support each other if and only if the corresponding <proposition> elements are annotated to support each other (`support` attribute).
            - Two nodes in the argument map attack each other if and only if the corresponding <proposition> elements are annotated to attack each other (`support` attribute).
        """)
            .strip()
            .format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)
        )

        if hints:
            prompt += "\n\n## Hints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
                "\n\n"
                "> [!WARNING]\n"
                "> For didactic purposes, I want you to make mistakes in your answer.\n"
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
        prompt = "Revise your previously submitted annotation and argument map given the above evaluation and feedback."

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
class ArgmapPlusArganno(Annotation, ArgumentMap):
    """
    Solution to the ArgmapPlusArganno problem: annotation and argdown snippet.
    
    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    annotated_source_text: str
    argdown_snippet: str
    unparsed_solution: str | None = None

    def __str__(self):
        if self.unparsed_solution:
            return self.unparsed_solution
        return self.annotated_source_text + "\n\n" + self.argdown_snippet

    @classmethod
    def from_raw_answer(
        cls, raw_answer: str
    ) -> "ArgmapPlusArganno":
        unparsed_solution = raw_answer
        annotated_source_text = ""
        argdown_snippet = ""
        if "```xml" in unparsed_solution:
            annotated_source_text = unparsed_solution.split("```xml")[-1].split("\n```")[0]
            annotated_source_text = "```xml" + annotated_source_text + "\n```"
        if "```argdown" in unparsed_solution:
            argdown_snippet = unparsed_solution.split("```argdown")[-1].split("\n```")[0]
            argdown_snippet = "```argdown" + argdown_snippet + "\n```"                

        return cls(
            annotated_source_text=annotated_source_text if annotated_source_text else unparsed_solution,
            argdown_snippet=argdown_snippet if argdown_snippet else unparsed_solution,
            unparsed_solution=None if annotated_source_text and argdown_snippet else unparsed_solution,
        )

class ArgmapPlusArgannoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusArgannoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + argument mapping problem must be a string or a list of strings"
        )


class ArgmapPlusArgannoJudge(Judge):
    """Judge for the anno plus argument mapping task."""

    def _evaluate_argmap(
        self, problem: ArgmapPlusArgannoProblem, reco: ArgmapPlusArganno
    ) -> Evaluation:
        is_valid = True
        artifacts: dict[str, Any] = {}
        eval_data = {
            "fenced_code_blocks": "",
            "annotation_nested_propositions": "",
            "annotation_missing_id": "",
            "annotation_duplicate_id": "",
            "annotation_invalid_support_ids": "",
            "annotation_invalid_attack_ids": "",
            "annotation_unknown_attributes": "",
            "argmap_invalid_argdown_syntax": "",
            "argmap_missing_labels": "",
            "argmap_duplicate_labels": "",
            "argmap_premise_conclusion_structures": "",
            "elements_correspondence": "",
            "relations_correspondence": "",
        }

        # check fenced codeblocks
        msgs = []
        ast = reco.annotated_source_text.strip("\n ")
        if not (ast.startswith("```xml") and ast.endswith("```")):
            msgs.append("Failed to extract fenced xml block with annotation.")
            if ast.count("```xml") == 0:
                msgs.append("No fenced code block starting with '```xml'.")
        ads = reco.argdown_snippet.strip("\n ")
        if not (ads.startswith("```argdown") and ads.endswith("```")):
            msgs.append("Failed to extract fenced argdown block.")
            if ads.count("```argdown") == 0:
                msgs.append("No fenced code block starting with '```argdown'.")
        if msgs:
            is_valid = False
            eval_data["fenced_code_blocks"] = " ".join(msgs)

        # evaluate anno
        evaluation_anno = AnnotationJudge()._evaluate_annotation(
            problem=AnnotationProblem(problem.sources, strip_html=False),
            annotation=Annotation(ast),
        )
        if evaluation_anno.is_valid is False:
            is_valid = False
        soup: BeautifulSoup = evaluation_anno.artifacts["soup"]
        artifacts["soup"] = soup
        for k, v in evaluation_anno.metrics.items():
            if k != "fenced_code_block":
                eval_data["annotation_" + k] = v

        # evaluate argmap
        evaluation_argmap = ArgMapJudge()._evaluate_argmap(
            problem=ArgMapProblem(problem.sources),
            argmap=ArgumentMap(ads),
        )
        if evaluation_argmap.is_valid is False:
            is_valid = False
        argdown: ArgdownMultiDiGraph = evaluation_argmap.artifacts["argdown_map"]
        artifacts["argdown_map"] = argdown
        for k, v in evaluation_argmap.metrics.items():
            if k != "fenced_code_block":
                eval_data["argmap_" + k] = v

        if argdown is None:
            return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)
        
        # check 1:1 correspondence between annotation and argmap elements
        msgs = []
        all_argmap_labels = [node.label for node in argdown.propositions + argdown.arguments if node.label]
        all_annotation_ids = [
            a.get("id") for a in soup.find_all("proposition") if a.get("id")  # type: ignore
        ]
        argument_label_map: dict[str,str] = {}
        for a in soup.find_all("proposition"):
            a_label = a.get("argument_label")  # type: ignore
            a_id = a.get("id")  # type: ignore
            if a_label not in all_argmap_labels:
                msgs.append(
                    f"Illegal 'argument_label' reference of proposition element with id={a_id}: "
                    f"No node with label '{a_label}' in the Argdown argument map."
                )
            else:
                argument_label_map[str(a_id)] = str(a_label)        

        for node in argdown.propositions + argdown.arguments:
            id_refs = node.data.get("annotation_ids", [])
            if not id_refs:
                msgs.append(
                    f"Missing 'annotation_ids' attribute of node with label '{node.label}'."
                )
                continue
            for id_ref in id_refs:
                if id_ref not in all_annotation_ids:
                    msgs.append(
                        f"Illegal 'annotation_ids' reference of node with label '{node.label}': "
                        f"No proposition element with id='{id_ref}' in the annotation."
                    )
                elif argument_label_map.get(id_ref) != node.label:
                    msgs.append(
                        f"Label reference mismatch: argument map node with label '{node.label}' "
                        f"has annotation_ids={str(id_refs)}, but the corresponding proposition element "
                        f"with id={id_ref} in the annotation has a different argument_label"
                        f"{': '+argument_label_map[id_ref] if id_ref in argument_label_map else ''}."
                    )
        if msgs:
            is_valid = False
            eval_data["elements_correspondence"] = " - ".join(msgs)
        del msgs

        # check support/attack relations correspondence
        msgs = []
        annotated_relations: list[dict] = []
        for a in soup.find_all("proposition"):
            from_id = a.get("id")  # type: ignore
            for support in a.get("supports", []):  # type: ignore
                if support in all_annotation_ids:
                    annotated_relations.append(
                        {
                            "from_id": from_id,
                            "to_id": support,
                            "valence": Valence.SUPPORT,
                        }
                    )
            for attacks in a.get("attacks", []):  # type: ignore
                if attacks in all_annotation_ids:
                    annotated_relations.append(
                        {
                            "from_id": from_id,
                            "to_id": attacks,
                            "valence": Valence.ATTACK,
                        }
                    )

        for ar in annotated_relations:
            if not any(
                dr.source == argument_label_map.get(ar["from_id"])
                and dr.target == argument_label_map.get(ar["to_id"])
                and dr.valence == ar["valence"]
                for dr in argdown.dialectical_relations
            ):
                msgs.append(
                    f"Annotated {str(ar['valence'])} relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by any relation in the argument map."
                )

        for dr in argdown.dialectical_relations:
            if not any(
                dr.source == argument_label_map.get(ar["from_id"])
                and dr.target == argument_label_map.get(ar["to_id"])
                and dr.valence == ar["valence"]
                for ar in annotated_relations
            ):
                msgs.append(
                    f"Dialectical {dr.valence.name} relation {dr.source} -> {dr.target} is not matched by any "
                    f"relation in the text annotation."
                )

        if msgs:
            is_valid = False
            eval_data["relations_correspondence"] = " - ".join(msgs)            

        return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgmapPlusArgannoProblem), "Problem must be an ArgmapPlusArgannoProblem"
        assert isinstance(original_solution, ArgmapPlusArganno) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgmapPlusArganno), (
                "All solutions must be ArgmapPlusArganno"
            )
            evaluations.append(self._evaluate_argmap(problem, solution))

        return evaluations


class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid solutions
    where the source text's annotated propositions are textually similiar to the node texts in the argument map."""

    hints = [
        "Make sure that your argument map stays faithful to and mimics closely "
        "the annotation of the source text. In particular, use a similar wording for claims as "
        "in the corresponding annotated source segments!"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:

        soup = evaluation.artifacts.get("soup")
        argdown = evaluation.artifacts.get("argdown_map")
        assert soup and argdown, (
            "AnnotationProximityPreferencePairGenerator: Missing soup or argdown in evaluation artifacts"
        )
        assert isinstance(soup, BeautifulSoup), "soup must be a BeautifulSoup object"
        assert isinstance(argdown, ArgdownMultiDiGraph), "argdown must be an ArgdownMultiDiGraph object"

        dlss: list[float] = []
        for anno_prop in soup.find_all("proposition"):
            anno_label = anno_prop.get("argument_label")  # type: ignore
            anno_text = anno_prop.get_text()  # type: ignore
            ad_prop = next((p for p in argdown.propositions if p.label==anno_label), None)
            if ad_prop and anno_text:
                for text in ad_prop.texts:
                    dlss.append(textdistance.damerau_levenshtein.normalized_similarity(text, anno_text))
            ad_arg = next((a for a in argdown.arguments if a.label==anno_label), None)
            if ad_arg and anno_text:
                for text in ad_arg.gists:
                    dlss.append(textdistance.damerau_levenshtein.normalized_similarity(text, anno_text))

        if not dlss:
            return 0.0
        
        return round(sum(dlss) / len(dlss), 1)
