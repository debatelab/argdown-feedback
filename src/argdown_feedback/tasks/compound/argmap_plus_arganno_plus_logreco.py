import dataclasses
from typing import Sequence

from textwrap import dedent
from bs4 import BeautifulSoup

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
)
from argdown_feedback.tasks.core.argmap import (
    ArgumentMap,
)
from argdown_feedback.tasks.core.arganno import (
    Annotation,
    AnnotationProblem,
    ANNOTATION_SCHEME,
)
from argdown_feedback.tasks.core.logreco import (
    LogicalReco,
)
from argdown_feedback.tasks.compound.argmap_plus_logreco import (
    ArgmapPlusLogrecoProblem,
)
from argdown_feedback.verifiers.base import BaseHandler, CompositeHandler
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    ArgmapLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.core.infreco_handler import (
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
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
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

            Analyse the argumentation in a given **source text**. Your answer is supposed to contain three artifacts:
            1. an argumentative text annotation,
            2. an Argdown argument map, and
            3. logical reconstructions of all the arguments in standard form (as deductively valid inferences).

            In the following, you find
            * the source text to analyse,
            * detailed instructions for how to annotate the source text (first artifact),
            * detailed instructions for how to create the Argdown argument map (second artifact),
            * detailed instructions for how to logically reconstruct and formalize the arguments in standard form (third artifact),
            * a description of how the three artifacts are supposed to cohere with each other,
            * formatting instructions for your answer.
            
            ## Source Text
                   
            ::: {{.source_text}}
            {sources}
            :::

            ## Annotation Task Details
                   
            Annotate the source text above according to the following schema:

            {annotation_scheme}

            Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                        
            Enclose the annotated text in a fenced codeblock, starting with '```xml {{filename="annotation.txt"}}' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                                      
            ## Argument Mapping Task Details
                   
            Create a syntactically correct Argdown argument map that captures the argumentation in the text. In particular, you should

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

            Importantly, enclose your Argdown argument map in a fenced codeblock, starting with '```argdown {{filename="map.ad"}}' and ending with '```'. If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Logical Argument Reconstruction Task Details                   

            Logically analyse and formally reconstruct the text's arguments with Argdown, ensuring the inferences are deductively valid.

            - Reconstruct *at least two arguments* in standard form (including premises, final conclusion, and possible intermediate conclusions).                   
            - For each proposition in your reconstruction (premises and conclusions), provide an adequate propositional logic / FOL formalization in NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Only declare variables that are used in the corresponding formalization and that have not been declared in the corresponding argument before. Ensure that your formalizations are consistent across different arguments.
            - For each inference step in the argument, provide information about which previously introduced premises or conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.
                  
            Importantly, enclose your Argdown reconstructions in a fenced codeblock, starting with '```argdown {{filename="reconstructions.ad"}}' and ending with '```'. If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

            ## Required Coherence of Annotation, Argument Map, and Argument Reconstructions

            The annotation, the argument map and your argument reconstructions must neatly correspond to each other. Meaning that:

            The argument reconstructions and the annotated source text must cohere with each other. Moreover, the inferential relations in the logically reconstructed arguments must reflect the annotated support relations. That is:
            
            1. Every argument in the argument map is reconstructed in standard form.
            2. Every reconstructed argument is present in the argument map.
            3. Every annotated text segment corresponds to a premise or conclusion in a reconstructed argument.
            4. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument -- and vice versa.
            5. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument -- and vice versa.
            6. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa.
            7. Whenever a claim A, in the _argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
            8. Whenever a text segment A in the _annotation_ supports another text segment B, then, in the _argdown reconstructions_, B's corresponding proposition is inferred from the proposition corresponding to A, or A refers to an argument that supports the argument referenced by B.
            9. Whenever a text segment A in the _annotation_ attacks another text segment B, then, in the _argdown reconstructions_, A's corresponding argument attacks the argument referenced by B.
            
            Here are the specific notation instructions which help you to ensure that annotation (first artifact), argument map (second artifact) and argument reconstructions (third artifact) fully cohere with each other in the above sense: 

            - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippets.
            - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding logically reconstructed argument.
            - Every premise and conclusion in the Argdown argument reconstructions has yaml inline data with an `annotation_ids` attribute that contains a (possibly empty) list of `id` attributes of the corresponding <proposition> elements in the annotation.
            - The argument labels in the argument map match (1-to-1) the argument labels in the argument reconstruction.
            - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstruction. 
            - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
            - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax).
            
            ## Formatting Recommendations
            
            To ensure that your submission is complete, it is recommended to format your answer as follows:
            
            ```xml {{filename="annotation.txt"}}
            <!-- annotated source text -->
            ```
            
            ```argdown {{filename="map.ad"}}
            // argument map
            ```
            
            ```argdown {{filename="reconstructions.ad"}}
            // formal argument reconstructions
            ```                   
            """)
            .strip()
            .format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)
        )

        if hints:
            prompt += "\n\n## Hints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

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
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

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
        return (
            self.annotated_source_text
            + "\n\n"
            + self.argdown_map_snippet
            + "\n\n"
            + self.argdown_reconstructions_snippet
        )

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusArgannoPlusLogreco":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)

        anno_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml
                and vr.code_snippet
                and vr.metadata
                and vr.metadata.get("filename") == "annotation.txt"
            ),
            None,
        )
        map_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown
                and vr.code_snippet
                and vr.metadata
                and vr.metadata.get("filename") == "map.ad"
            ),
            None,
        )
        reco_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown
                and vr.code_snippet
                and vr.metadata
                and vr.metadata.get("filename") == "reconstructions.ad"
            ),
            None,
        )

        return cls(
            annotated_source_text=anno_snippet if anno_snippet else raw_answer,
            argdown_map_snippet=map_snippet if map_snippet else raw_answer,
            argdown_reconstructions_snippet=reco_snippet
            if reco_snippet
            else raw_answer,
            unparsed_solution=None if map_snippet and reco_snippet else raw_answer,
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


class ArgmapPlusArgannoPlusLogrecoJudge(MPJudge):
    """Judge for the argmap plus infreco task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgmapPlusArgannoPlusLogrecoProblem), (
            "Problem must be an ArgmapPlusArgannoPlusLogrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusArgannoPlusLogreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgmapPlusArgannoPlusLogreco) for solution in solutions
        ), "All solutions must be ArgmapPlusArgannoPlusLogreco objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgmapPlusArgannoPlusLogrecoProblem), "Problem must be an ArgmapPlusArgannoPlusLogrecoProblem"
        assert isinstance(solution, ArgmapPlusArgannoPlusLogreco), "Solution must be an ArgmapPlusArgannoPlusLogreco"

        anno_filter = BaseHandler.create_metadata_filter("filename", ["annotation.txt"])
        map_filter = BaseHandler.create_metadata_filter("filename", ["map.ad"])
        reco_filter = BaseHandler.create_metadata_filter(
            "filename", ["reconstructions.ad"]
        )

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasAtLeastNArgumentsHandler(name="InfReco.HasAtLeastNArgumentsHandler",filter=reco_filter, N=2),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=reco_filter),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=reco_filter),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=reco_filter),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=reco_filter),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=reco_filter),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", filter=reco_filter),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", filter=reco_filter),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", filter=reco_filter),
                # Extra material handlers
                NoExtraPropositionsHandler(name="InfReco.NoExtraPropositionsHandler", filter=reco_filter),
            ]
        )
        main_handler = CompositeHandler(
            handlers=[
                # Processing
                DefaultProcessingHandler(),
                HasAnnotationsHandler(filter=anno_filter),
                HasArgdownHandler(name="HasArgdownHandler.map", filter=map_filter),
                HasArgdownHandler(name="HasArgdownHandler.reco", filter=reco_filter),
                # Core
                ArgannoCompositeHandler(filter=anno_filter),
                ArgMapCompositeHandler(filter=map_filter),
                infreco_handler,
                LogRecoCompositeHandler(filter=reco_filter),
                # Coherence
                ArgannoInfrecoCoherenceHandler(),
                ArgmapInfrecoCoherenceHandler(),
                ArgmapLogrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = main_handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation


