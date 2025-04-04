import copy
from typing import Any, Sequence

import dataclasses
from textwrap import dedent
from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    Argument,
    Conclusion,
    Proposition,
    Valence,
    DialecticalType,
    parse_argdown,
)

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_hirpo.tasks.core.argmap import (
    ArgMapJudge,
    ArgMapProblem,
    ArgumentMap,
    ConnectednessPreferencePairGenerator,
    MaxArgsPreferencePairGenerator,
    MaxSupportsPreferencePairGenerator,
    MaxAttacksPreferencePairGenerator,
    SourceTextProximityPreferencePairGenerator,
)
from argdown_hirpo.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco,
)
from argdown_hirpo.verifiers.infreco_verifier import InfRecoVerifier

NEGATION_SCHEMES = [
    "NOT: {text}",
    "Not: {text}",
    "NOT {text}",
    "Not {text}",   
]

class ArgmapPlusInfrecoProblem(InfRecoProblem, ArgMapProblem):
    """Task: Create coherent informal reco and argument map."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
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
            # Assignment: Present a text's argumentation as an informal Argdown argument map, and reconstruct its arguments in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create two coherent Argdown code snippets: One with an informal argument map, and another one with reconstructions of all the arguments in standard form (premise-conclusion structure).

            ::: {{.source_text}}              
            {sources}
            :::

                   
            ## Argument Mapping Task Details                   
                   
            Create a syntactically correct informal Argdown argument map that reconstructs the argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

            Importantly, enclose your Argdown argument map in a fenced codeblock, starting with '```argdown {{filename="map.ad"}}' and ending with '```'. If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.


            ## Argument Reconstruction Task Details                   

            Informally analyse and reconstruct the text's arguments with Argdown. In particular, you should

            - reconstruct *at least two arguments* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).
            - provide, for each conclusion in an argument, information about which previously introduced premises or 
              conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
            - ensure that every premise and intermdeiate conclusions is actually used to infer a conclusion in the argument.
                  
            Importantly, enclose your Argdown reconstructions in a fenced codeblock, starting with '```argdown {{filename="reconstructions.ad"}}' and ending with '```'. If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

                   
            ## Required Coherence of Annotation and Argument Reconstruction                                            

            The argument map and your argument reconstructions must neatly correspond to each other. Meaning that:
                   
            1. Every argument in the argument map is reconstructed in standard form.
            2. Every reconstructed argument is present in the argument map.
            3. Whenever a claim in the argument map supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument -- and vice versa.
            4. Whenever an argument in the argument map supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument -- and vice versa.
            5. Whenever an argument A in the argument map supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa.
                   
            Here are the specific notation instructions which help you to ensure that argument map and argument reconstructions fully cohere with each other in the above sense: 

            - The argument labels in the argument map must match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstruction. 
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label or, absent any label, have string-identical texts.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if they have different labels, and one text prepends "NOT: " the other text. (Avoid double negations and rely on duplex negatio affirmat instead.)
        """)
            .strip()
            .format(sources=self.sources)
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
        prompt = "Revise your previously submitted argument map and argument reconstructions given the above evaluation and feedback."

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
class ArgmapPlusInfreco(Solution):
    """
    Solution to the ArgmapPlusInfreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    argdown_map_snippet: str
    argdown_reconstructions_snippet: str
    unparsed_solution: str | None = None

    def __str__(self):
        if self.unparsed_solution:
            return self.unparsed_solution
        return self.argdown_map_snippet + "\n\n" + self.argdown_reconstructions_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusInfreco":
        unparsed_solution = raw_answer
        argdown_map_snippet = ""
        argdown_reconstructions_snippet = ""

        _code_label = 'argdown {filename="map.ad"}'
        if f"\n```{_code_label}" in unparsed_solution:
            argdown_map_snippet = unparsed_solution.split(f"\n```{_code_label}")[-1].split("\n```")[0]
            argdown_map_snippet = f"```{_code_label}" + argdown_map_snippet + "\n```"

        _code_label = 'argdown {filename="reconstructions.ad"}'
        if f"\n```{_code_label}" in unparsed_solution:
            argdown_reconstructions_snippet = unparsed_solution.split(f"\n```{_code_label}")[-1].split("\n```")[0]
            argdown_reconstructions_snippet = f"```{_code_label}" + argdown_reconstructions_snippet + "\n```"

        return cls(
            argdown_map_snippet=argdown_map_snippet
                                if argdown_map_snippet
                                else unparsed_solution,
            argdown_reconstructions_snippet=argdown_reconstructions_snippet
                                if argdown_reconstructions_snippet
                                else unparsed_solution,
            unparsed_solution=None
                                if argdown_map_snippet and argdown_reconstructions_snippet
                                else unparsed_solution,
        )

    def partial_argmap(self) -> ArgumentMap:
        """Return the argument map subsolution."""
        return ArgumentMap(
            argdown_snippet=self.argdown_map_snippet,
        )

    def partial_infreco(self) -> InformalReco:
        """Return the informal reconstruction subsolution."""
        return InformalReco(
            argdown_snippet=self.argdown_reconstructions_snippet,
        )


class ArgmapPlusInfrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusInfrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgmapPlusInfrecoJudge(Judge):
    """Judge for the argmap plus infreco task."""

    @staticmethod
    def are_identical(prop1: Proposition | None, prop2: Proposition | None) -> bool:
        """Check if two propositions are identical."""
        if prop1 is None or prop2 is None:
            return False
        return (
            prop1.label == prop2.label
            or any(text in prop2.texts for text in prop1.texts)
        )

    @staticmethod
    def are_contradictory(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
        """Check if two propositions are identical."""
        if prop1 is None or prop2 is None:
            return False
        if prop1.label == prop2.label:
            return False
        if argdown is not None:
            if any(
                drel.source in [prop1.label, prop2.label]
                and drel.target in [prop1.label, prop2.label]
                and drel.source != drel.target
                and drel.valence in [Valence.ATTACK, Valence.CONTRADICT]
                for drel in argdown.dialectical_relations
            ):
                return True
        negations_prop1 = [
            scheme.format(text=text) for scheme in NEGATION_SCHEMES for text in prop1.texts
        ]
        negations_prop2 = [
            scheme.format(text=text) for scheme in NEGATION_SCHEMES for text in prop2.texts
        ]
        return (
            any(text in negations_prop2 for text in prop1.texts)
            or any(text in negations_prop1 for text in prop2.texts)
        )
    
    @staticmethod
    def indirectly_supports(from_label: str, to_label: str, argdown_map: Argdown) -> bool:
        """Check if one node directly or indirectly (via intermediate prop) supports another one in argument map."""
        if from_label == to_label:
            return True

        rels_direct = argdown_map.get_dialectical_relation(from_label, to_label)
        rels_direct = [] if rels_direct is None else rels_direct
        if any(rd.valence == Valence.SUPPORT for rd in rels_direct):
            return True

        for prop in argdown_map.propositions:
            if prop.label is None or prop.label == from_label or prop.label == to_label:
                continue
            rels1 = argdown_map.get_dialectical_relation(from_label, prop.label)
            rels2 = argdown_map.get_dialectical_relation(prop.label, to_label)
            rels1 = [] if rels1 is None else rels1
            rels2 = [] if rels2 is None else rels2
            for rel1 in rels1:
                for rel2 in rels2:
                    if (
                        (rel1.valence == Valence.SUPPORT and rel2.valence == Valence.SUPPORT)
                        or (rel1.valence in [Valence.ATTACK, Valence.CONTRADICT] and rel2.valence in [Valence.ATTACK, Valence.CONTRADICT]) 
                    ):
                        return True

        return False


    @staticmethod
    def indirectly_attacks(from_label: str, to_label: str, argdown_map: Argdown) -> bool:
        """Check if one node directly or indirectly (via intermediate prop) attacks another one in argument map."""
        if from_label == to_label:
            return False

        rels_direct = argdown_map.get_dialectical_relation(from_label, to_label)
        rels_direct = [] if rels_direct is None else rels_direct
        if any(rd.valence == Valence.ATTACK for rd in rels_direct):
            return True

        for prop in argdown_map.propositions:
            if prop.label is None or prop.label == from_label or prop.label == to_label:
                continue
            rels1 = argdown_map.get_dialectical_relation(from_label, prop.label)
            rels2 = argdown_map.get_dialectical_relation(prop.label, to_label)
            rels1 = [] if rels1 is None else rels1
            rels2 = [] if rels2 is None else rels2
            for rel1 in rels1:
                for rel2 in rels2:
                    if (
                        (rel1.valence == Valence.SUPPORT and rel2.valence in [Valence.ATTACK, Valence.CONTRADICT])
                        or (rel1.valence in [Valence.ATTACK, Valence.CONTRADICT] and rel2.valence == Valence.SUPPORT) 
                    ):
                        return True

        return False

    def _evaluate_coherence(self, argdown_map: Argdown, argdown_reco: Argdown) -> dict[str, str]:
        """Evaluate the coherence between the argument map and the informal reconstruction."""

        eval_data: dict[str, str] = {}
        
        # check elements correspondence
        msgs = []
        map_labels = list(set(a.label for a in argdown_map.arguments))
        reco_labels = list(set(a.label for a in argdown_reco.arguments))
        for label in map_labels:
            if label not in reco_labels:
                msgs.append(f"Argument <{label}> in map is not reconstructed (argument label mismatch).")
        for label in reco_labels:
            if label not in map_labels:
                msgs.append(f"Reconstructed argument <{label}> is not in the map (argument label mismatch).")            
        map_prop_labels = list(set(p.label for p in argdown_map.propositions))
        reco_prop_labels = list(set(p.label for p in argdown_reco.propositions))
        for label in map_prop_labels:
            if label not in reco_prop_labels:
                msgs.append(f"Claim [{label}] in argument map has no corresponding proposition in reconstructions (proposition label mismatch).")
        if msgs:
            eval_data["elements_correspondence"] = " - ".join(msgs)

        # check relations correspondence
        msgs = []
        for drel in argdown_map.dialectical_relations:
            if DialecticalType.SKETCHED not in drel.dialectics:
                continue
            # get matched source nodes in reco
            source_m: Argument | Proposition | None
            target_m: Argument | Proposition | None
            if any(a.label==drel.source for a in argdown_map.arguments):
                source_m =  next( 
                    (a for a in argdown_reco.arguments if a.label==drel.source),
                    None
                )  
            else:
                source_m = next(
                    (p for p in argdown_reco.propositions if p.label==drel.source),
                    None
                )
            if any(a.label==drel.target for a in argdown_map.arguments):
                target_m = next(
                    (a for a in argdown_reco.arguments if a.label==drel.target),
                    None
                )
            else:
                target_m = next(
                    (p for p in argdown_reco.propositions if p.label==drel.target),
                    None
                )
            if source_m is None or target_m is None:
                continue
            # check if the relation is grounded in reco
            if isinstance(source_m, Argument) and isinstance(target_m, Argument):
                if not source_m.pcs or not target_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if any(
                        self.are_identical(
                            argdown_reco.get_proposition(pr.proposition_label),
                            argdown_reco.get_proposition(source_m.pcs[-1].proposition_label)
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from <{drel.source}> to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, conclusion of <{drel.source}> does "
                        f"not figure as premise in <{drel.target}>."
                    )
                elif drel.valence == Valence.ATTACK:
                    if any(
                        self.are_contradictory(
                            argdown_reco.get_proposition(pr.proposition_label),
                            argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                            argdown_reco
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from <{drel.source}> to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, conclusion of <{drel.source}> does "
                        f"not contradict any premise in <{drel.target}>."
                    )
            if isinstance(source_m, Proposition) and isinstance(target_m, Argument):
                if not target_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if any(
                        self.are_identical(
                            argdown_reco.get_proposition(pr.proposition_label),
                            source_m,
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from [{drel.source}] to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.source}] does "
                        f"not figure as premise in <{drel.target}>."
                    )
                elif drel.valence == Valence.ATTACK:
                    if any(
                        self.are_contradictory(
                            argdown_reco.get_proposition(pr.proposition_label),
                            source_m,
                            argdown_reco
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from [{drel.source}] to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.source}] does "
                        f"not contradict any premise in <{drel.target}>."
                    )
            if isinstance(source_m, Argument) and isinstance(target_m, Proposition):
                if not source_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if self.are_identical(
                        argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                        target_m,
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from <{drel.source}> to [{drel.target}] in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.target}] "
                        f"does not figure as conclusion in <{drel.source}>."
                    )
                if drel.valence == Valence.ATTACK:
                    if self.are_contradictory(
                        argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                        target_m,
                        argdown_reco
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from <{drel.source}> to [{drel.target}] in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.target}] "
                        f"does not contradict the conclusion of <{drel.source}>."
                    )

        if msgs:
            eval_data["relations_correspondence"] = " - ".join(msgs)

        return eval_data

    def _evaluate_solution(
        self, problem: ArgmapPlusInfrecoProblem, reco: ArgmapPlusInfreco
    ) -> Evaluation:
        is_valid = True
        artifacts: dict[str, Any] = {}
        eval_data = {
            "fenced_code_blocks": "",

            "argmap_invalid_argdown_syntax": "",
            "argmap_missing_labels": "",
            "argmap_duplicate_labels": "",
            "argmap_premise_conclusion_structures": "",

            "recos_invalid_argdown_syntax": "",
            "recos_no_arguments": "",
            "recos_illformed_arguments": "",  # starts with conclusion / ends with premise / no pcs
            "recos_missing_inference_info": "",
            "recos_unknown_proposition_references": "",  # in inference info
            "recos_unused_propositions": "",

            "elements_correspondence": "",
            "relations_correspondence": "",
        }

        # check fenced codeblocks
        msgs = []
        _code_label = 'argdown {filename="map.ad"}'
        ad_map = reco.argdown_map_snippet.strip("\n ")
        if not (ad_map.startswith(f"```{_code_label}") and ad_map.endswith("```")):
            msgs.append("Failed to extract fenced xml block with annotation.")
            if ad_map.count(f"```{_code_label}") == 0:
                msgs.append(f"No fenced code block starting with '```{_code_label}'.")
        _code_label = 'argdown {filename="reconstructions.ad"}'
        ad_reco = reco.argdown_reconstructions_snippet.strip("\n ")
        if not (ad_reco.startswith(f"```{_code_label}") and ad_reco.endswith("```")):
            msgs.append("Failed to extract fenced argdown block.")
            if ad_reco.count(f"```{_code_label}") == 0:
                msgs.append(f"No fenced code block starting with '```{_code_label}'.")
        if msgs:
            eval_data["fenced_code_blocks"] = " ".join(msgs)
        del msgs

        # evaluate argmap
        evaluation_argmap = ArgMapJudge()._evaluate_argmap(
            problem=ArgMapProblem(sources=problem.sources),
            argmap=ArgumentMap(ad_map.replace('```argdown {filename="map.ad"}', '```argdown')),
        )
        argdown_map: ArgdownMultiDiGraph = evaluation_argmap.artifacts["argdown_map"]
        artifacts["argdown_map"] = argdown_map
        for k, v in evaluation_argmap.metrics.items():
            if k != "fenced_code_block":
                eval_data["argmap_" + k] = v


        # evaluate argdown reco
        if ad_reco.startswith("```argdown") and ad_reco.endswith("```"):
            ad_reco = "\n".join(ad_reco.splitlines()[1:-1])
        try:
            argdown_reco = parse_argdown(ad_reco)
        except Exception as e:
            argdown_reco = None
            eval_data["recos_invalid_argdown_syntax"] = (
                f"Failed to parse argdown: {str(e)}"
            )

        artifacts["argdown_reco"] = argdown_reco
        if argdown_reco:

            if len(argdown_reco.arguments) == 0:
                eval_data["recos_no_arguments"] = "No argument in argdown snippet."

            eval_dimensions_map = copy.deepcopy(InfRecoVerifier.default_eval_dimensions_map)        
            print(eval_dimensions_map)    
            eval_dimensions_map["illformed_argument"].remove("has_not_multiple_gists")
            eval_dimensions_map["missing_label_gist"].remove("has_gist")
            eval_dimensions_map["disallowed_material"].remove("only_grounded_dialectical_relations")
            eval_dimensions_map["disallowed_material"].remove("no_extra_propositions")
            reco_eval_data = InfRecoVerifier.run_battery(argdown_reco, eval_dimensions_map)
            print(f"Reco eval data: {reco_eval_data}")
            for k,v in reco_eval_data.items():
                eval_data["recos_" + k] = v


        # evaluate coherence between argmap and reco
        if argdown_map and argdown_reco:
            coherence_eval_data = self._evaluate_coherence(
                argdown_map = argdown_map,
                argdown_reco = argdown_reco,
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
        assert isinstance(problem, ArgmapPlusInfrecoProblem), (
            "Problem must be an ArgannoPlusInfrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusInfreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgmapPlusInfreco), (
                "All solutions must be ArgmapPlusInfreco"
            )
            evaluations.append(self._evaluate_solution(problem, solution))

        return evaluations


class SimplicityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the ArgmapPlusInfreco, prefering valid reconstructions
    with succinct and simple propositions."""

    hints = [
        "Make sure that you keep each of the arguments premises and conclusion(s) simple and succinct. "
        "Short sentences are crucial at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        reco: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_reco" in evaluation.artifacts, (
            "Evaluation must contain argdown_reco artifact"
        )
        argdown_reco: ArgdownMultiDiGraph = evaluation.artifacts["argdown_reco"]
        propositions: list[Proposition] = argdown_reco.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) ** -1 if lengths else 0


class ConnectednessPreferencePairGeneratorCT(ConnectednessPreferencePairGenerator):
    """Simple wrapper around ConnectednessPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(is_valid=True, artifacts={"argdown_map": argdown}, metrics={}),
        )
    
class MaxArgsPreferencePairGeneratorCT(MaxArgsPreferencePairGenerator):
    """Simple wrapper around MaxArgsPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide partial argmap"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(is_valid=True, artifacts={"argdown_map": argdown}, metrics={}),
        )
    

class MaxSupportsPreferencePairGeneratorCT(MaxSupportsPreferencePairGenerator):
    """Simple wrapper around MaxSupportsPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(is_valid=True, artifacts={"argdown_map": argdown}, metrics={}),
        )
    

class MaxAttacksPreferencePairGeneratorCT(MaxAttacksPreferencePairGenerator):
    """Simple wrapper around MaxAttacksPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(is_valid=True, artifacts={"argdown_map": argdown}, metrics={}),
        )
    
class SourceTextProximityPreferencePairGeneratorCT(SourceTextProximityPreferencePairGenerator):
    """Simple wrapper around SourceTextProximityPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert "argdown_map" in evaluation.artifacts, (
            "Evaluation must contain argdown_map artifact"
        )
        assert hasattr(solution, "partial_argmap"), (
            "Solution must provide a partial_argmap method"
        )
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        return super()._score(
            problem=problem,
            argmap=solution.partial_argmap(),
            evaluation=Evaluation(is_valid=True, artifacts={"argdown_map": argdown}, metrics={}),
        )
    
