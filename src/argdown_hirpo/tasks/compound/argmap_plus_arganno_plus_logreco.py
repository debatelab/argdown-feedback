import copy
import dataclasses
from typing import Sequence

from textwrap import dedent
from bs4 import BeautifulSoup

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
)
from argdown_hirpo.tasks.compound.arganno_plus_infreco import ArgannoPlusInfrecoJudge
from argdown_hirpo.tasks.core.argmap import (
    ArgumentMap,
)
from argdown_hirpo.tasks.core.arganno import (
    Annotation,
    AnnotationJudge,
    AnnotationProblem,
    ANNOTATION_SCHEME,
)
from argdown_hirpo.tasks.core.logreco import (
    LogicalReco,
)
from argdown_hirpo.tasks.compound.argmap_plus_logreco import (
    ArgmapPlusLogreco,
    ArgmapPlusLogrecoJudge,
    ArgmapPlusLogrecoProblem,
)


class ArgmapPlusArgannoPlusLogrecoProblem(ArgmapPlusLogrecoProblem, AnnotationProblem):
    """Task: Create coherent annotation, logical reco and argument map."""

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
            # Assignment: Annotate a source text, present its argumentation as an informal Argdown argument map, and logically reconstruct its arguments in standard form using Argdown syntax.
                        
            Analyse the argumentation in the following **source text**. Create three coherent code artifacts: an argumentative text annotation, an informal argument map, and logical reconstructions of all the arguments in standard form (as deductively valid inferences).

            ::: {{.source_text}}              
            {sources}
            :::

                   
            ## Annotation Task Details                   
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way.
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml {{filename="annotation.txt"}}' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.

                                      
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

                   
            ## Required Coherence of Annotation, Argument Map, and Argument Reconstructions                                            

            The annotation, the argument map and your argument reconstructions must neatly correspond to each other. Meaning that:

            The argument reconstruction and the annotated source text must cohere with each other.    Moreover, the inferential relations in the reconstructed argument should reflect the annotated support relations.
                   
            1. Every argument in the argument map is reconstructed in standard form.
            2. Every reconstructed argument is present in the argument map.
            3. Every annotated text segment corresponds to a premise or conclusion in a reconstructed argument.
            4. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument -- and vice versa.
            5. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument -- and vice versa.
            6. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa.
            7. Whenever a claim A, in the _argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
            8. Whenever a text segment A in the _annotation_ supports another text segment B, then B corresponds to a conclusion in the reconstructions which is directly or indirectly supported by a proposition corresponding to A.
                                      
            Here are the specific notation instructions which help you to ensure that annotation, argument map and argument reconstructions fully cohere with each other in the above sense: 

            - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippets.
            - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding argument reconstruction. 
            - Every premise and conclusion in the Argdown argument reconstructions has yaml inline data with an `annotation_ids` attribute that contains a (possibly empty) list of `id` attributes of the corresponding <proposition> elements in the annotation.
            - The argument labels in the argument map match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstruction. 
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax).
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
        prompt = "Revise your previously submitted annotation, argument map, and argument reconstructions given the above evaluation and feedback."

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
class ArgmapPlusArgannoPlusLogreco(Solution):
    """
    Solution to the ArgmapPlusArgannoPlusLogreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    annotated_source_text: str
    argdown_map_snippet: str
    argdown_reconstructions_snippet: str
    unparsed_solution: str | None = None

    def __str__(self):
        if self.unparsed_solution:
            return self.unparsed_solution
        return self.annotated_source_text + "\n\n" + self.argdown_map_snippet + "\n\n" + self.argdown_reconstructions_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusArgannoPlusLogreco":
        unparsed_solution = raw_answer
        annotated_source_text = ""
        argdown_map_snippet = ""
        argdown_reconstructions_snippet = ""

        _code_label = 'xml {filename="annotation.txt"}'
        if f'\n```{_code_label}' in unparsed_solution:
            annotated_source_text = unparsed_solution.split("\n```xml")[-1].split("\n```")[0]
            annotated_source_text = f"```{_code_label}" + annotated_source_text + "\n```"

        _code_label = 'argdown {filename="map.ad"}'
        if f"\n```{_code_label}" in unparsed_solution:
            argdown_map_snippet = unparsed_solution.split(f"\n```{_code_label}")[-1].split("\n```")[0]
            argdown_map_snippet = f"```{_code_label}" + argdown_map_snippet + "\n```"

        _code_label = 'argdown {filename="reconstructions.ad"}'
        if f"\n```{_code_label}" in unparsed_solution:
            argdown_reconstructions_snippet = unparsed_solution.split(f"\n```{_code_label}")[-1].split("\n```")[0]
            argdown_reconstructions_snippet = f"```{_code_label}" + argdown_reconstructions_snippet + "\n```"

        return cls(
            annotated_source_text=annotated_source_text
                                if annotated_source_text
                                else unparsed_solution,
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

    def partial_annotation(self) -> Annotation:
        """Return the annotation subsolution."""
        return Annotation(
            annotated_source_text=self.annotated_source_text,
        )

    def partial_argmap(self) -> ArgumentMap:
        """Return the argument map subsolution."""
        return ArgumentMap(
            argdown_snippet=self.argdown_map_snippet,
        )

    def partial_logreco(self) -> LogicalReco:
        """Return the informal reconstruction subsolution."""
        return LogicalReco(
            argdown_snippet=self.argdown_reconstructions_snippet,
        )


class ArgmapPlusArgannoPlusLogrecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgmapPlusArgannoPlusLogrecoProblem(inputs)
        raise ValueError(
            "Inputs to an annotation + infreco problem must be a string or a list of strings"
        )


class ArgmapPlusArgannoPlusLogrecoJudge(ArgmapPlusLogrecoJudge):
    """Judge for the argmap plus infreco task."""

    def _evaluate_solution(
        self, problem: Problem, reco: Solution
    ) -> Evaluation:
        
        # evaluate argmap+logreco
        evaluation = super()._evaluate_solution(problem, reco)  # type: ignore
        argdown_reco = evaluation.artifacts["argdown_reco"]
        artifacts = copy.deepcopy(evaluation.artifacts)
        eval_data = evaluation.metrics.copy()
        is_valid = evaluation.is_valid

        eval_data["argdown_elements_correspondence"] = eval_data.pop("elements_correspondence", "")
        eval_data["argdown_relations_correspondence"] = eval_data.pop("relations_correspondence", "")

        eval_data.update({
            "annotation_empty": "",
            "annotation_nested_propositions": "",
            "annotation_missing_id": "",
            "annotation_duplicate_id": "",
            "annotation_invalid_support_ids": "",
            "annotation_invalid_attack_ids": "",
            "annotation_unknown_attributes": "",
            "annotation_elements_correspondence": "",
            "annotation_relations_correspondence": "",
        })

        # check fenced annotation codeblocks
        msgs = [eval_data["fenced_code_blocks"]]
        _code_label = 'xml {filename="annotation.txt"}'
        ast = reco.argdown_map_snippet.strip("\n ")  # type: ignore
        if not (ast.startswith(f"```{_code_label}") and ast.endswith("```")):
            msgs.append('Failed to extract fenced xml block with annotation.')
            if ast.count(f"```{_code_label}") == 0:
                msgs.append(f"No fenced code block starting with '```{_code_label}'.")
        if msgs:
            eval_data["fenced_code_blocks"] = " ".join(msgs)
        del msgs

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
            eval_data["annotation_empty"] = "No proposition elements in the annotation."

        # evaluate coherence annotation<>reco
        coherence_eval_data = ArgannoPlusInfrecoJudge()._evaluate_coherence(
            soup_anno = soup,
            argdown_reco = argdown_reco,
        )
        coherence_eval_data = {
            "annotation_"+k: v
            for k, v in coherence_eval_data.items()
        }
        eval_data.update(coherence_eval_data)
                
        is_valid = is_valid and not any(v for v in eval_data.values())        

        return Evaluation(is_valid=is_valid, artifacts=artifacts, metrics=eval_data)


    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgmapPlusArgannoPlusLogrecoProblem), (
            "Problem must be an ArgannoPlusLogRecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusArgannoPlusLogreco)
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

