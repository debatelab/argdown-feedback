import dataclasses
from typing import Sequence

from textwrap import dedent
from pyargdown import (
    ArgdownMultiDiGraph,
)
import textdistance

from argdown_hirpo.tasks.base import (
    Judge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_hirpo.logic.fol_to_nl import FOL2NLTranslator
from argdown_hirpo.tasks.core.logreco import (
    LogicalReco,
)
from argdown_hirpo.tasks.compound.argmap_plus_infreco import (
    ArgmapPlusInfreco,
    ArgmapPlusInfrecoProblem,
)
from argdown_hirpo.verifiers.base import BaseHandler, CompositeHandler
from argdown_hirpo.verifiers.coherence.argmap_logreco_handler import ArgmapLogrecoCoherenceHandler
from argdown_hirpo.verifiers.coherence.argmap_infreco_handler import ArgmapInfrecoCoherenceHandler
from argdown_hirpo.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_hirpo.verifiers.core.content_check_handler import (
    HasArgdownHandler,
)
from argdown_hirpo.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasAtLeastNArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    NoExtraPropositionsHandler,
    UsesAllPropsHandler,
)
from argdown_hirpo.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_hirpo.verifiers.processing_handler import (
    DefaultProcessingHandler,
)
from argdown_hirpo.verifiers.verification_request import (
    VerificationRequest,
)




class ArgmapPlusLogrecoProblem(ArgmapPlusInfrecoProblem):
    """Task: Create coherent logical reco and argument map."""

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
            # Assignment: Present a text's argumentation as an informal Argdown argument map, and logically reconstruct its arguments in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create two coherent Argdown code snippets: One with an informal argument map, and another one with logical reconstructions of all the arguments in standard form (as deductively valid inferences).

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

            Logically analyse and reconstruct the text's arguments with Argdown, ensuring the inferences are deductively valid.

            - Reconstruct *at least two arguments* in standard form (including premises, final 
              conclusion, and possible intermediate conclusions).
                   
            - For each proposition in your reconstruction (premises and conclusions), provide an adequate FOL formalization in
              NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses.
              Only declare variables that are used in the corresponding formalization and that have not been declared in the 
              corresponding argument before. Ensure that your formalizations are consistent across different arguments.

            - For each inference step in the argument, provide information about which previously introduced premises or 
              conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
              where the list items refer to the respective premise or conclusion labels.
                  
            Importantly, enclose your Argdown reconstructions in a fenced codeblock, starting with '```argdown {{filename="reconstructions.ad"}}' and ending with '```'. If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

                   
            ## Required Coherence of Annotation and Argument Reconstruction                                            

            The argument map and your argument reconstructions must neatly correspond to each other. Meaning that:
                   
            1. Every argument in the argument map is reconstructed in standard form.
            2. Every reconstructed argument is present in the argument map.
            3. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument -- and vice versa.
            4. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument -- and vice versa.
            5. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa.
            6. Whenever a claim A, in the _argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
                   
            Here are the specific notation instructions which help you to ensure that argument map and argument reconstructions fully cohere with each other in the above sense: 

            - The argument labels in the argument map must match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstruction. 
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax).
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
class ArgmapPlusLogreco(ArgmapPlusInfreco):
    """
    Solution to the ArgmapPlusLogreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    def partial_logreco(self) -> LogicalReco:
        """Return the informal reconstruction subsolution."""
        return LogicalReco(
            argdown_snippet=self.argdown_reconstructions_snippet,
        )


class ArgmapPlusLogrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusLogrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgmapPlusLogrecoJudge(Judge):
    """Judge for the argmap plus infreco task."""

    # @staticmethod
    # def are_identical(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
    #     """Check if two propositions are identical."""
    #     if (
    #         prop1 is None or prop1.label is None
    #         or prop2 is None or prop2.label is None
    #     ):
    #         return False

    #     # equivalence via dialectical relations        
    #     if argdown is not None:
    #         rels1 = argdown.get_dialectical_relation(prop1.label, prop2.label)
    #         rels2 = argdown.get_dialectical_relation(prop2.label, prop1.label)
    #         if rels1 and rels2 :
    #             for rel1 in rels1:
    #                 for rel2 in rels2:
    #                     if (
    #                         rel1 is not None and rel2 is not None
    #                         and rel1.valence == Valence.SUPPORT
    #                         and DialecticalType.AXIOMATIC in rel1.dialectics
    #                         and rel2.valence == Valence.SUPPORT
    #                         and DialecticalType.AXIOMATIC in rel2.dialectics
    #                     ):
    #                         return True

    #     return prop1.label == prop2.label

    # @staticmethod
    # def are_contradictory(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
    #     """Check if two propositions are identical."""
    #     if prop1 is None or prop2 is None:
    #         return False
    #     if prop1.label == prop2.label:
    #         return False
    #     if argdown is not None:
    #         if any(
    #             drel.source in [prop1, prop2]
    #             and drel.target in [prop1, prop2]
    #             and drel.source != drel.target
    #             and drel.valence in [Valence.ATTACK, Valence.CONTRADICT]
    #             and DialecticalType.AXIOMATIC in drel.dialectics
    #             for drel in argdown.dialectical_relations
    #         ):
    #             return True
    #     return False


    # def _evaluate_coherence(self, argdown_map: Argdown, argdown_reco: Argdown) -> dict[str, str]:
    #     """Evaluate the coherence between the argument map and the informal reconstruction."""

    #     eval_data: dict[str, str] = {}
        
    #     # check elements correspondence
    #     #print("ARGMAP<>LOGRECO check")
    #     msgs = []
    #     map_alabels = list(set(a.label for a in argdown_map.arguments))
    #     reco_alabels = list(set(a.label for a in argdown_reco.arguments))
    #     #print(f"Map alabels: {map_alabels}")
    #     #print(f"Reco plabels: {reco_alabels}")
    #     for label in map_alabels:
    #         if label not in reco_alabels:
    #             msgs.append(f"Argument <{label}> in map is not reconstructed (argument label mismatch).")
    #     for label in reco_alabels:
    #         if label not in map_alabels:
    #             msgs.append(f"Reconstructed argument <{label}> is not in the map (argument label mismatch).")            
    #     map_prop_labels = list(set(p.label for p in argdown_map.propositions))
    #     reco_prop_labels = list(set(p.label for p in argdown_reco.propositions))
    #     for label in map_prop_labels:
    #         if label not in reco_prop_labels:
    #             msgs.append(f"Claim [{label}] in argument map has no corresponding proposition in reconstructions (proposition label mismatch).")
    #     if msgs:
    #         eval_data["elements_correspondence"] = " - ".join(msgs)

    #     # check relations correspondence
    #     msgs = []

    #     for drel in argdown_map.dialectical_relations:
    #         #print(f"Checking if {drel} in argmap is grounded in reco...")
    #         if drel.source not in reco_alabels+reco_prop_labels or drel.target not in reco_alabels+reco_prop_labels:
    #             #print(f"Skipping {drel} in argmap, labels not in reco.")
    #             continue
    #         if DialecticalType.SKETCHED in drel.dialectics:
    #             rel_matches = argdown_reco.get_dialectical_relation(drel.source, drel.target)
    #             rel_matches = [] if rel_matches is None else rel_matches
    #             #print(f"Found potential matches: {rel_matches}")

    #             if any(
    #                 rm.valence == drel.valence
    #                 and DialecticalType.GROUNDED in rm.dialectics
    #                 for rm in rel_matches
    #             ):
    #                 continue

    #             if not any(rm.valence == drel.valence for rm in rel_matches):
    #                 msgs.append(
    #                     f"Dialectical {drel.valence.name} relation from node '{drel.source}' to node '{drel.target}' "
    #                     f"in argument map is not matched by any relation in the argument reconstruction."
    #                 )
    #                 continue
    #             msgs.append(
    #                 f"Dialectical {drel.valence.name} relation from node '{drel.source}' to node '{drel.target}' "
    #                 f"in argument map is not grounded in logical argument reconstructions."
    #             )


    #     for drel in argdown_reco.dialectical_relations:
    #         if drel.source not in map_alabels+map_prop_labels or drel.target not in map_alabels+map_prop_labels:
    #             continue
    #         if DialecticalType.GROUNDED in drel.dialectics:
    #             if drel.valence == Valence.SUPPORT:
    #                 if not ArgmapPlusLogrecoJudge.indirectly_supports(drel.source, drel.target, argdown_map):
    #                     msgs.append(
    #                         f"According to the argument reconstructions, item '{drel.source}' supports item '{drel.target}', "
    #                         f"but this dialectical relation is not captured in the argument map."
    #                     )
    #             elif drel.valence == Valence.ATTACK:
    #                 if not ArgmapPlusLogrecoJudge.indirectly_attacks(drel.source, drel.target, argdown_map):
    #                     msgs.append(
    #                         f"According to the argument reconstructions, item '{drel.source}' attacks item '{drel.target}', "
    #                         f"but this dialectical relation is not captured in the argument map."
    #                     )

    #     if msgs:
    #         eval_data["relations_correspondence"] = " - ".join(msgs)

    #     return eval_data

    def _evaluate_solution(
        self, problem: ArgmapPlusLogrecoProblem, solution: ArgmapPlusLogreco
    ) -> Evaluation:
        

        map_filter = BaseHandler.create_metadata_filter(
            "filename", ["map.ad"]
        )
        reco_filter = BaseHandler.create_metadata_filter(
            "filename", ["reconstructions.ad"]
        )

        infreco_handler = InfRecoCompositeHandler(
            handlers = [
                # Argument existence handlers
                HasAtLeastNArgumentsHandler(filter=reco_filter,N=2),
                HasPCSHandler(filter=reco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(filter=reco_filter),
                EndsWithConclusionHandler(filter=reco_filter),
                NoDuplicatePCSLabelsHandler(filter=reco_filter),
                # Label and gist handlers
                HasLabelHandler(filter=reco_filter),
                # Inference data handlers
                HasInferenceDataHandler(filter=reco_filter),
                PropRefsExistHandler(filter=reco_filter),
                UsesAllPropsHandler(filter=reco_filter),
                # Extra material handlers
                NoExtraPropositionsHandler(filter=reco_filter),
            ]    
        )
        main_handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasArgdownHandler(filter=map_filter),
                HasArgdownHandler(filter=reco_filter),
                ArgMapCompositeHandler(filter=map_filter),
                infreco_handler,
                LogRecoCompositeHandler(filter=reco_filter),
                ArgmapInfrecoCoherenceHandler(),
                ArgmapLogrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=str(solution), source=problem.sources
        )
        result = main_handler.handle(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation


        # TODO remove
        # assert isinstance(problem, ArgmapPlusLogrecoProblem), (
        #     "Problem must be an ArgmapPlusLogrecoProblem"
        # )
        # # check that reco has 'argdown_map_snippet' and 'argdown_reconstructions_snippet' attributes
        # assert hasattr(reco, "argdown_map_snippet"), (
        #     "Solution must have 'argdown_map_snippet' attribute"
        # )
        # assert hasattr(reco, "argdown_reconstructions_snippet"), (
        #     "Solution must have 'argdown_reconstructions_snippet' attribute"
        # )
        # is_valid = True
        # artifacts: dict[str, Any] = {}
        # eval_data = {
        #     "fenced_code_blocks": "",

        #     "argmap_invalid_argdown_syntax": "",
        #     "argmap_missing_labels": "",
        #     "argmap_duplicate_labels": "",
        #     "argmap_premise_conclusion_structures": "",

        #     "recos_invalid_argdown_syntax": "",
        #     "recos_too_few_arguments": "",
        #     "recos_illformed_arguments": "",  # starts with conclusion / ends with premise / no pcs
        #     "recos_missing_inference_info": "",
        #     "recos_unknown_proposition_references": "",  # in inference info
        #     "recos_unused_propositions": "",
        #     "recos_disallowed_material": "",  # more propositions
        #     "recos_flawed_formalizations": "",
        #     "recos_invalid_inference": "",
        #     "recos_redundant_premises": "",
        #     "recos_inconsistent_premises": "",
        #     "recos_formally_ungrounded_relations": "",

        #     "elements_correspondence": "",
        #     "relations_correspondence": "",
        # }

        # # check fenced codeblocks
        # msgs = []
        # _code_label = 'argdown {filename="map.ad"}'
        # ad_map = reco.argdown_map_snippet.strip("\n ")
        # if not (ad_map.startswith(f"```{_code_label}") and ad_map.endswith("```")):
        #     msgs.append("Failed to extract fenced xml block with annotation.")
        #     if ad_map.count(f"```{_code_label}") == 0:
        #         msgs.append(f"No fenced code block starting with '```{_code_label}'.")
        # _code_label = 'argdown {filename="reconstructions.ad"}'
        # ad_reco = reco.argdown_reconstructions_snippet.strip("\n ")
        # if not (ad_reco.startswith(f"```{_code_label}") and ad_reco.endswith("```")):
        #     msgs.append("Failed to extract fenced argdown block.")
        #     if ad_reco.count(f"```{_code_label}") == 0:
        #         msgs.append(f"No fenced code block starting with '```{_code_label}'.")
        # if msgs:
        #     eval_data["fenced_code_blocks"] = " ".join(msgs)
        # del msgs

        # # evaluate argmap
        # evaluation_argmap = ArgMapJudge()._evaluate_argmap(
        #     problem=ArgMapProblem(sources=problem.sources),
        #     argmap=ArgumentMap(ad_map.replace('```argdown {filename="map.ad"}', '```argdown')),
        # )
        # argdown_map: ArgdownMultiDiGraph = evaluation_argmap.artifacts["argdown_map"]
        # artifacts["argdown_map"] = argdown_map
        # for k, v in evaluation_argmap.metrics.items():
        #     if k != "fenced_code_block":
        #         eval_data["argmap_" + k] = v


        # # evaluate argdown reco
        # if ad_reco.startswith("```argdown") and ad_reco.endswith("```"):
        #     ad_reco = "\n".join(ad_reco.splitlines()[1:-1])
        # try:
        #     argdown_reco = parse_argdown(ad_reco)
        # except Exception as e:
        #     argdown_reco = None
        #     eval_data["recos_invalid_argdown_syntax"] = (
        #         f"Failed to parse argdown: {str(e)}"
        #     )

        # artifacts["argdown_reco"] = argdown_reco
        # if argdown_reco:

        #     if len(argdown_reco.arguments) < 2:
        #         eval_data["recos_too_few_arguments"] = "Too few arguments in argdown snippet (at least 2 required)."

        #     eval_dimensions_map = copy.deepcopy(LogRecoVerifier.default_eval_dimensions_map)        
        #     print(eval_dimensions_map)    
        #     eval_dimensions_map["missing_label_gist"].remove("has_gist")
        #     eval_dimensions_map.pop("disallowed_material")
        #     reco_eval_data, all_expressions, all_declarations = LogRecoVerifier.run_battery(argdown_reco, eval_dimensions_map)
        #     print(f"Reco eval data: {reco_eval_data}")
        #     for k,v in reco_eval_data.items():
        #         eval_data["recos_" + k] = v

        #     check, msg = LogRecoVerifier._no_extra_propositions(argdown_reco)
        #     if check is False:
        #         eval_data["recos_disallowed_material"] = (
        #             msg
        #             if msg
        #             else "Some propositions are not used as premise and/or conclusion."
        #         )

        #     artifacts["all_expressions"] = all_expressions
        #     artifacts["all_declarations"] = all_declarations

        #     # check for formally_ungrounded_relations
        #     check, msg = LogRecoVerifier._has_formally_grounded_relations(
        #         argdown_reco=argdown_reco,
        #         all_expressions=all_expressions,
        #         all_declarations=all_declarations,
        #     )
        #     if check is False:
        #         eval_data["recos_formally_ungrounded_relations"] = (
        #             msg
        #             if msg
        #             else "Some dialectical relations between propositions are not grounded in their logical formalizations."
        #         )

        # # evaluate coherence between argmap and reco
        # if argdown_map and argdown_reco:
        #     coherence_eval_data = self._evaluate_coherence(
        #         argdown_map = argdown_map,
        #         argdown_reco = argdown_reco,
        #     )
        #     eval_data.update(coherence_eval_data)
                
        # is_valid = not any(v for v in eval_data.values())

        # return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgmapPlusLogrecoProblem), (
            "Problem must be an ArgannoPlusLogRecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusLogreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgmapPlusLogreco), (
                "All solutions must be ArgmapPlusLogreco"
            )
            evaluations.append(self._evaluate_solution(problem, solution))

        return evaluations


class GlobalFormalizationsFaithfulnessPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Global FormalizationsFaithfulnessPreferencePairGenerator"""
    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown_reco = evaluation.artifacts.get("argdown_reco")
        assert argdown_reco is not None and isinstance(argdown_reco, ArgdownMultiDiGraph), (
            "Evaluation must contain argdown_reco artifact"
        )
        all_expressions = evaluation.artifacts.get("all_expressions")
        assert all_expressions is not None and isinstance(all_expressions, dict), (
            "Evaluation must contain all_expressions artifact"
        )
        all_declarations = evaluation.artifacts.get("all_declarations")
        assert all_declarations is not None and isinstance(all_declarations, dict), (
            "Evaluation must contain all_declarations artifact"
        )

        dlds: list[float] = []
        for argument in argdown_reco.arguments:
            print(f"Argument: {argument.label}")
            for pr in argument.pcs:
                expression = all_expressions.get(pr.proposition_label)
                if expression is None:
                    continue

                proposition = argdown_reco.get_proposition(pr.proposition_label)
                if proposition is None:
                    continue

                text_1 = FOL2NLTranslator.translate_to_nl_sentence(
                    expression, all_declarations
                )
                print(f"Text 1: {text_1}")

                for text_2 in proposition.texts:
                    print(f"Text 2: {text_2}")
                    dlds.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text_1, text_2
                        )
                    )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0