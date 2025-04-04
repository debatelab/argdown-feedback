from typing import Any, Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import (
    Argdown,
    Conclusion,
    Valence,
    parse_argdown,
    Argument,
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
from argdown_hirpo.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco,
)
from argdown_hirpo.verifiers.infreco_verifier import InfRecoVerifier


# utility #


def _get_props_used_in_inference(
    argument: Argument, pr_label: str, from_key: str = "from"
) -> list[str]:
    """Get all proposition labels used directly or indirectly in the inference
    to a conclusion with label `pr_label`."""

    if argument is None or not argument.pcs:
        return []

    used_labels = set()

    def add_parent_labels(label: str):
        c = next(
            (c for c in argument.pcs if isinstance(c, Conclusion) and c.label == label),
            None,
        )
        if c is None:
            return []
        parent_labels = c.inference_data.get(from_key, [])
        used_labels.update(parent_labels)
        for ref in parent_labels:
            add_parent_labels(ref)

    add_parent_labels(pr_label)

    return list(used_labels)


class ArgannoPlusInfrecoProblem(InfRecoProblem, AnnotationProblem):
    """Task: Create coherent informal reco and arg annotation."""

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
            # Assignment: Annotate a source text and reconstruct its main argument in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create a a coherent argumentative text annotation and a corresponding informal argument reconstruction in standard form (premise-conclusion structure).

            ::: {{.source_text}}              
            {sources}
            :::

            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way.
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                   
            ## Argument Reconstruction Task Details                   

            Informally analyse and reconstruct the text's main argumentation with Argdown. In particular, you should

            - reconstruct *at least one argument* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).
            - provide, for each conclusion in an argument, information about which previously introduced premises or 
              conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
                  
            Importantly, enclose your Argdown snippet in a fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation and Argument Reconstruction                                                

            The argument reconstruction and the annotated source text must cohere with each other. There should be a one-to-many correspondence between premises/conclusion(s) and annotated text segments. Moreover, the inferential relations in the reconstructed argument should reflect the annotated support relations.
                   
            In particular, you should ensure that: 

            - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippet.
            - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding argument. 
            - Every premise and conclusion in the Argdown argument has yaml inline data with an `annotation_ids` attribute that contains a list of `id` attributes of the corresponding <proposition> elements in the annotation.
            - If, in the annotation, one <proposition> element supports another one (via its `support` attribute), then, in the Argdown argument, the proposition corresponding to the former element is used to infer the conclusion corresponding to the latter element.
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
        prompt = "Revise your previously submitted annotation and argument reconstruction given the above evaluation and feedback."

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
class ArgannoPlusInfreco(Annotation, InformalReco):
    """
    Solution to the ArgannoPlusInfreco problem: annotation and argdown snippet.

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
    def from_raw_answer(cls, raw_answer: str) -> "ArgannoPlusInfreco":
        unparsed_solution = raw_answer
        annotated_source_text = ""
        argdown_snippet = ""
        if "```xml" in unparsed_solution:
            annotated_source_text = unparsed_solution.split("```xml")[-1].split(
                "\n```"
            )[0]
            annotated_source_text = "```xml" + annotated_source_text + "\n```"
        if "```argdown" in unparsed_solution:
            argdown_snippet = unparsed_solution.split("```argdown")[-1].split(
                "\n```"
            )[0]
            argdown_snippet = "```argdown" + argdown_snippet + "\n```"

        return cls(
            annotated_source_text=annotated_source_text
            if annotated_source_text
            else unparsed_solution,
            argdown_snippet=argdown_snippet if argdown_snippet else unparsed_solution,
            unparsed_solution=None
            if annotated_source_text and argdown_snippet
            else unparsed_solution,
        )


class ArgannoPlusInfrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgannoPlusInfrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgannoPlusInfrecoJudge(Judge):
    """Judge for the anno plus argument mapping task."""

    def _evaluate_coherence(self, soup_anno: BeautifulSoup, argdown_reco: Argdown) -> dict[str, str]:
        """Check the coherence between the annotation and the argument reconstruction."""
        eval_data = {
            "elements_correspondence": "",
            "relations_correspondence": "",
        }
        msgs = []
        all_argument_labels = [arg.label for arg in argdown_reco.arguments if arg.label]
        all_annotation_ids = [
            a.get("id")  # type: ignore
            for a in soup_anno.find_all("proposition")
            if a.get("id")  # type: ignore
        ]
        argument_label_map: dict[str, str] = {}
        proposition_label_map: dict[str, str] = {}
        refreco_map: dict[str, str] = {}
        for a in soup_anno.find_all("proposition"):
            a_label = a.get("argument_label")  # type: ignore
            a_id = a.get("id")  # type: ignore
            a_ref_reco = a.get("ref_reco_label")  # type: ignore
            if a_label not in all_argument_labels:
                msgs.append(
                    f"Illegal 'argument_label' reference of proposition element with id={a_id}: "
                    f"No argument with label '{a_label}' in the Argdown snippet."
                )
                continue
            argument_label_map[str(a_id)] = str(a_label)
            refreco_map[str(a_id)] = str(a_ref_reco)
            argument = next(arg for arg in argdown_reco.arguments if arg.label == a_label)
            if not argument.pcs or not any(
                a_ref_reco == pr.label for pr in argument.pcs
            ):
                msgs.append(
                    f"Illegal 'ref_reco_label' reference of proposition element with id={a_id}: "
                    f"No premise or conclusion with label '{a_ref_reco}' in argument '{a_label}'."
                )
            else:
                pr = next(
                    pr for pr in argument.pcs if a_ref_reco == pr.label
                )
                proposition = next(
                    prop
                    for prop in argdown_reco.propositions
                    if prop.label == pr.proposition_label
                )
                proposition_label_map[str(a_id)] = str(pr.proposition_label)
                id_refs = proposition.data.get("annotation_ids", [])
                if str(a_id) not in id_refs:
                    msgs.append(
                        f"Label reference mismatch: proposition element with id={a_id} in the annotation "
                        f"references (via ref_reco) the proposition '{pr.label}' of argument '{argument.label}', "
                        f"but the annotation_ids={str(id_refs)} of that proposition do not include the id={a_id}."
                    )

        for argument in argdown_reco.arguments:
            if argument.label not in argument_label_map.values():
                msgs.append(
                    f"Free floating argument: Argument '{argument.label}' does not have any "
                    "corresponding elements in the annotation."
                )
            for pr in argument.pcs:
                proposition = next(
                    prop
                    for prop in argdown_reco.propositions
                    if prop.label == pr.proposition_label
                )
                id_refs = proposition.data.get("annotation_ids")
                if id_refs is None:
                    msgs.append(
                        f"Missing 'annotation_ids' attribute in proposition '{pr.label}' "
                        f"of argument '{argument.label}'."
                    )
                    continue
                for id_ref in id_refs:
                    if id_ref not in all_annotation_ids:
                        msgs.append(
                            f"Illegal 'annotation_ids' reference in proposition '{pr.label}' of argument '{argument.label}': "
                            f"No proposition element with id='{id_ref}' in the annotation."
                        )
                        continue
                    """
                    if argument_label_map.get(id_ref) != argument.label:
                        msgs.append(
                            f"Label reference mismatch: proposition '{pr.label}' of argument '{argument.label}' "
                            f"has annotation_ids={str(id_refs)}, but the corresponding proposition element "
                            f"with id={id_ref} in the annotation has a different argument_label"
                            f"{': ' + argument_label_map[id_ref] if id_ref in argument_label_map else ''}."
                        )
                    if refreco_map.get(id_ref) != pr.label:
                        msgs.append(
                            f"Label reference mismatch: proposition '{pr.label}' of argument '{argument.label}' "
                            f"has annotation_ids={str(id_refs)}, but the corresponding proposition element "
                            f"with id={id_ref} in the annotation has a different ref_reco_label"
                            f"{': ' + refreco_map[id_ref] if id_ref in refreco_map else ''}."
                        )
                    """

        for i in range(len(argdown_reco.propositions)):
            for j in range(i + 1, len(argdown_reco.propositions)):
                prop1 = argdown_reco.propositions[i]
                prop2 = argdown_reco.propositions[j]
                dps = [
                    f"'{x}'"
                    for x in prop1.data.get("annotation_ids", [])
                    if x in prop2.data.get("annotation_ids", [])
                ]
                if dps:
                    msgs.append(
                        f"Label reference mismatch: annotation text segment(s) {', '.join(dps)} "
                        f"are referenced by distinct propositions in the Argdown argument "
                        f"reconstruction ('{prop1.label}', '{prop2.label}')."
                    )

        if msgs:
            eval_data["elements_correspondence"] = " - ".join(msgs)
        del msgs

        # check support and attack relations correspondence
        msgs = []
        annotated_support_relations: list[dict] = []
        annotated_attack_relations: list[dict] = []
        for a in soup_anno.find_all("proposition"):
            from_id = a.get("id")  # type: ignore
            for support in a.get("supports", []):  # type: ignore
                if support in all_annotation_ids:
                    annotated_support_relations.append(
                        {
                            "from_id": str(from_id),
                            "to_id": str(support),
                        }
                    )
            for attack in a.get("attacks", []):  # type: ignore
                if attack in all_annotation_ids:
                    annotated_attack_relations.append(
                        {
                            "from_id": str(from_id),
                            "to_id": str(attack),
                        }
                    )

        # helper
        def _drel_fn(x,y):
            drels = argdown_reco.get_dialectical_relation(x, y)
            return drels if drels is not None else []

        for ar in annotated_support_relations:
            arglabel_from = argument_label_map.get(ar["from_id"])
            proplabel_from = proposition_label_map.get(ar["from_id"])
            arglabel_to = argument_label_map.get(ar["to_id"])
            proplabel_to = proposition_label_map.get(ar["to_id"])
            if arglabel_from is None or arglabel_to is None:
                msgs.append(
                    f"Annotated support relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by any relation in the reconstruction (illegal argument_labels)."
                )
                continue
            if arglabel_from != arglabel_to:
                drels = _drel_fn(
                    arglabel_from, arglabel_to,
                ) + _drel_fn(
                    arglabel_from, proplabel_to,
                ) + _drel_fn(
                    proplabel_from, arglabel_to,
                ) + _drel_fn(
                    proplabel_from, proplabel_to,
                )
                if drels is None or not any(
                    dr.valence == Valence.SUPPORT
                    for dr in drels
                ):
                    msgs.append(
                        f"Proposition elements {ar['from_id']} and {ar['to_id']} are annotated to support each other, but "
                        f"none of the corresponding Argdown elements <{arglabel_from}>/[{proplabel_from}] supports "
                        f"<{arglabel_to}> or [{proplabel_to}]."
                    )
                continue
            del argument
            argument = next(
                (arg for arg in argdown_reco.arguments if arg.label == arglabel_from),
                None  # type: ignore
            )
            ref_reco_from = refreco_map.get(ar["from_id"])
            ref_reco_to = refreco_map.get(ar["to_id"])
            if argument is None or ref_reco_from is None or ref_reco_to is None:
                continue
            if ref_reco_from not in _get_props_used_in_inference(argument, ref_reco_to):
                msgs.append(
                    f"Annotated support relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by the inferential relations in the argument '{argument.label}'."
                )

        for ar in annotated_attack_relations:
            arglabel_from = argument_label_map.get(ar["from_id"])
            proplabel_from = proposition_label_map.get(ar["from_id"])
            arglabel_to = argument_label_map.get(ar["to_id"])
            proplabel_to = proposition_label_map.get(ar["to_id"])
            if arglabel_from is None or arglabel_to is None:
                msgs.append(
                    f"Annotated attack relation from {ar['from_id']} to {ar['to_id']} is not "
                    f"matched by any relation in the reconstruction (illegal argument_labels)."
                )
                continue
            if arglabel_from == arglabel_to:
                msgs.append(
                    f"Text segments assigned to the same argument cannot attack each other "
                    f"({ar['from_id']} attacks {ar['to_id']} while both are assigned to {arglabel_from})."
                )
                continue
            drels = _drel_fn(
                arglabel_from, arglabel_to,
            ) + _drel_fn(
                arglabel_from, proplabel_to,
            ) + _drel_fn(
                proplabel_from, arglabel_to,
            ) + _drel_fn(
                proplabel_from, proplabel_to,
            )
            if drels is None or not any(
                dr.valence == Valence.ATTACK
                for dr in drels
            ):
                msgs.append(
                    f"Proposition elements {ar['from_id']} and {ar['to_id']} are annotated to attack each other, but "
                    f"none of the corresponding Argdown elements <{arglabel_from}>/[{proplabel_from}] attacks "
                    f"<{arglabel_to}> or [{proplabel_to}]."
                )
        if msgs:
            eval_data["relations_correspondence"] = " - ".join(msgs)
        del msgs

        return eval_data

    def _evaluate_solution(
        self, problem: Problem, reco: Solution
    ) -> Evaluation:
        is_valid = True
        artifacts: dict[str, Any] = {}
        eval_data = {
            "fenced_code_blocks": "",
            "annotation_empty": "",
            "annotation_nested_propositions": "",
            "annotation_missing_id": "",
            "annotation_duplicate_id": "",
            "annotation_invalid_support_ids": "",
            "annotation_invalid_attack_ids": "",
            "annotation_unknown_attributes": "",
            "argument_invalid_argdown_syntax": "",
            "argument_missing": "",
            "argument_illformed_argument": "",  # starts with conclusion / ends with premise
            "argument_missing_inference_info": "",
            "argument_unknown_proposition_references": "",  # in inference info
            "elements_correspondence": "",
            "relations_correspondence": "",
        }

        # check fenced codeblocks
        msgs = []
        ast = reco.annotated_source_text.strip("\n ")  # type: ignore
        if not (ast.startswith("```xml") and ast.endswith("```")):
            msgs.append("Failed to extract fenced xml block with annotation.")
            if ast.count("```xml") == 0:
                msgs.append("No fenced code block starting with '```xml'.")
        ads = reco.argdown_snippet.strip("\n ")  # type: ignore
        if not (ads.startswith("```argdown") and ads.endswith("```")):
            msgs.append("Failed to extract fenced argdown block.")
            if ads.count("```argdown") == 0:
                msgs.append("No fenced code block starting with '```argdown'.")
        if msgs:
            is_valid = False
            eval_data["fenced_code_blocks"] = " ".join(msgs)

        # evaluate anno
        evaluation_anno = AnnotationJudge()._evaluate_annotation(
            problem=AnnotationProblem(problem.sources, strip_html=False),  # type: ignore
            annotation=Annotation(ast),
        )
        if evaluation_anno.is_valid is False:
            is_valid = False
        soup: BeautifulSoup = evaluation_anno.artifacts["soup"]
        artifacts["soup"] = soup
        for k, v in evaluation_anno.metrics.items():
            if k != "fenced_code_block":
                eval_data["annotation_" + k] = v
        if not list(soup.find_all("proposition")):
            is_valid = False
            eval_data["annotation_empty"] = "No proposition elements in the annotation."

        # evaluate argdown reco
        if ads.startswith("```argdown") and ads.endswith("```"):
            ads = "\n".join(ads.splitlines()[1:-1])
        try:
            argdown = parse_argdown(ads)
        except Exception as e:
            argdown = None
            is_valid = False
            eval_data["argument_invalid_argdown_syntax"] = (
                f"Failed to parse argdown: {str(e)}"
            )

        artifacts["argdown"] = argdown
        if argdown is None:
            return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)

        if len(argdown.arguments) == 0:
            is_valid = False
            eval_data["argument_missing"] = "No argument in argdown snippet."

        # any illformed arguments
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = InfRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.has_pcs()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' lacks premise conclusion structure: {msg}"
                )
            check, msg = verifier.starts_with_premise()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' does not start with a premise: {msg}"
                )

            check, msg = verifier.ends_with_conclusion()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' does not end with a conclusion: {msg}"
                )

            check, msg = verifier.has_not_multiple_gists()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' has more than one gist: {msg}"
                )

            check, msg = verifier.has_no_duplicate_pcs_labels()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' has duplicate labels in the standard form: {msg}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_illformed_argument"] = " ".join(msgs)
        del msgs

        # any missing inference info
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = InfRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.has_inference_data()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' lacks inference information: {msg}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_missing_inference_info"] = " ".join(msgs)

        # any unknown proposition references
        msgs = []
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = InfRecoVerifier(argdown, argument_idx=argument_idx)
            check, msg = verifier.prop_refs_exist()
            if check is False:
                msgs.append(
                    f"Argument '{argument.label}' has unknown proposition references: {msg}"
                )
        if msgs:
            is_valid = False
            eval_data["argument_unknown_proposition_references"] = " ".join(msgs)
        del msgs

        # check 1:1 correspondence between annotation and argument elements

        coherence_eval_data = self._evaluate_coherence(
            soup_anno = soup,
            argdown_reco = argdown,
        )
        eval_data.update(coherence_eval_data)
                
        is_valid = not any(v for v in eval_data.values())        

        return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgannoPlusInfrecoProblem), (
            "Problem must be an ArgannoPlusInfrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgannoPlusInfreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgannoPlusInfreco), (
                "All solutions must be ArgannoPlusInfreco"
            )
            evaluations.append(self._evaluate_solution(problem, solution))

        return evaluations


class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid solutions
    where the source text's annotated propositions are textually similiar to the node texts in the argument map."""

    hints = [
        (
            "Make sure that your argument reconstruction stays faithful to and mimics closely "
            "the annotation of the source text. In particular, use formulations of premises and conclusions "
            "that are similar to the corresponding annotated text segments!"
        )
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:

        soup = evaluation.artifacts["soup"]
        anno_props = soup.find_all("proposition")

        argdown = evaluation.artifacts["argdown"]

        matches: list[tuple[str, str]] = []
        for proposition in argdown.propositions:
            for annotation_id in proposition.data.get("annotation_ids", []):
                anno_prop = next(
                    (ap for ap in anno_props if ap.get("id") == annotation_id), None
                )
                if anno_prop is None:
                    continue
                for text in proposition.texts:
                    matches.append((anno_prop.get_text(), text))

        print("matches")
        print(matches)
        dlss = [
            textdistance.damerau_levenshtein.normalized_similarity(s, t)
            for s, t in matches
        ]
        return round(sum(dlss) / len(dlss), 1)
