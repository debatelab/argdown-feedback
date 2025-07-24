import random
from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
import textdistance

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    ScoringVirtuePreferencePairGenerator,
)
from argdown_feedback.tasks.core.arganno import (
    ANNOTATION_SCHEME,
    Annotation,
    AnnotationProblem,
)
from argdown_feedback.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco,
)
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
    HasAnnotationsHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)
from argdown_feedback.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)


ARGANNO_PLUS_INFRECO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    # Assignment: Annotate a source text and informally reconstruct its main argument in standard form using Argdown syntax.
           
    Analyse the argumentation in the given **source text**. Your submission is supposed to contain two artifacts:
    1. an argumentative text annotation and
    2. an Argdown snippet with informal reconstructions of the main argumentation in standard form (premise-conclusion-structure).

    In the following, you find
    * the source text to analyse,
    * detailed instructions for how to annotate the source text (first artifact),
    * detailed instructions for how to informally reconstruct the argumentation (second artifact),
    * a description of how both artifacts are supposed to cohere with each other,
    * formatting instructions for your answer.
    
    ## Source Text
           
    ::: {{.source_text}}
    {sources}
    :::

    ## Annotation Task Details                   
           
    Annotate the source text above according to the following schema:

    {annotation_scheme}

    Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                
    Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
    
    ## Argument Reconstruction Task Details                   

    Informally analyse and reconstruct the text's main argumentation with Argdown. In particular, you should

    - reconstruct *at least one argument* in standard form (including the argument label, premises, final conclusion, and possible intermediate conclusions).
    - provide, for each conclusion in every argument reconstructed, information about which previously introduced premises or conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.

    Importantly, enclose all your reconstructions, separated by newlines, in a single fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

    ## Required Coherence of Annotation and Argument Reconstruction                                                

    Your source text annotation (first artifact) and your argument reconstruction (second artifact) must cohere with each other. There should be a one-many correspondence between premises/conclusion(s) in the Argdown arguments and marked-up elements in the text annotation. Moreover, the inferential relations in the reconstructed argument should reflect the annotated support relations.

    In particular, you should ensure that:

    - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippet.
    - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding argument. 
    - Every premise and conclusion in the Argdown argument has yaml inline data with an `annotation_ids` attribute that contains a (possibly empty) list of `id` attributes of the corresponding <proposition> elements in the annotation.
    - If, in the annotation, one <proposition> element supports another one (via its `supports` attribute), then, in the Argdown argument, the proposition corresponding to the former element is used to infer the conclusion corresponding to the latter element.

    Please submit your answer below, containing both appropriately formatted artifacts enclosed in separate code blocks, e.g.:

    ```xml
    <!-- your annotated source text here -->
    ```

    ```argdown
    /* your Argdown snippet here */
    ```
    """).strip(),
    # Elementary school style
    dedent("""
    # Let's Be Argument Detectives! 🔍🔎

    Hello there! Today we have a special mission. We need to find the hidden arguments in a text and show how they work. We'll do this in TWO fun steps!

    ## The Text We're Investigating

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Mark Up the Text! 🖍️

    First, we need to mark all the important parts of the text that make arguments. Use these special tags:

    {annotation_scheme}

    Here's what to do:
    1. Find all parts that make claims or give reasons
    2. Put <proposition> tags around them
    3. Give each one a special ID number
    4. Show which parts support other parts
    5. Connect each part to our argument breakdown in Step 2

    Put your marked-up text in a special box like this:
    ```xml
    (your marked text goes here)
    ```

    ## Step 2: Build the Argument! 🏗️

    Now, let's put together at least one complete argument from the text. Show:
    1. All the reasons (premises)
    2. The main conclusion
    3. Any steps in between

    For each conclusion, show which reasons it comes from using, for example:
    `-- {{'from': ['2','4']}} --`

    Put your argument in another special box:
    ```argdown
    (your argument goes here)
    ```

    ## SUPER IMPORTANT: Connect Both Parts! 🔗

    The most important job is making sure your marked text and your argument match perfectly:

    1. Each marked part needs an `argument_label` showing which argument it belongs to
    2. Each marked part needs a `ref_reco_label` showing which premise or conclusion it is
    3. Each premise and conclusion needs `annotation_ids` showing which text parts it came from
    4. If parts support each other in the text, they should also connect in the argument

    ## EXAMPLE

    Let us flesh this out with an example, shall we?
           
    ```xml
    <proposition id="i1" argument_label="Fun Course" ref_reco_label="1" supports="i2">This course is fun!</proposition> We are having a great time. <proposition id="i2" argument_label="Fun Course" ref_reco_label="3">I'll recommend it to my friends.</proposition>
    ```

    ```argdown
    <Fun Course>

    (1) This course is fun. {{annotation_ids: ['i1']}}
    (2) My friends love fun courses. {{annotation_ids: []}}
    -- {{'from': ['1', '2']}} --
    (3) I will recommend this course to my friends. {{annotation_ids: ['i2']}} 
    ```

    I know you'll be an AMAZING argument detective! Let's solve this case! 🕵️‍♀️🕵️‍♂️
    """).strip(),
    # Casual/friendly style
    dedent("""
    # Hey there! Let's break down an argument together

    I've got this text that I need analyzed in two complementary ways - first by marking up the argumentative parts directly in the text, and then by reconstructing the main argument(s) in a clear, structured format. Could you help me out with this?

    ## Here's the text we're working with:

    ::: {{.source_text}}
    {sources}
    :::

    ## First task: Annotate the text

    I need you to mark up the text to show which parts function as reasons. You'll use these XML tags:

    {annotation_scheme}

    Just add the tags around the important bits - don't change the wording or anything. If there are long sections that don't contain arguments, you can shorten those parts.

    When you're done, put your annotated text in a code block that starts with ```xml and ends with ```.

    ## Second task: Reconstruct the argument(s)

    Next, take the arguments you identified and reconstruct them in standard form using Argdown. I need you to:
    - Start each argument with a label (e.g., `<My Argument>`), followed by an empty line
    - Include all the premises (supporting reasons)
    - Show the main conclusion
    - Add any intermediate conclusions if needed
    - For each conclusion, note which premises it follows from using this format: `-- {{'from': ['1','2','3']}} --`

    Put all these reconstructions (separated by new lines) in another single code block starting with ```argdown and ending with ```.

    ## Important: Connect your annotation with your reconstruction

    The key part is making sure your annotation and reconstruction work together:
    - Each proposition in your annotation should have an `argument_label` pointing to the argument it belongs to
    - Each proposition should also have a `ref_reco_label` pointing to its corresponding premise/conclusion
    - Each premise and conclusion in your reconstruction should have `annotation_ids` showing which text parts it came from (may be an empty list)
    - If something supports something else in your annotation, it should also support it in your reconstruction

    Thanks for your help with this! Looking forward to seeing how you break down the argument.
    """).strip(),
    # Academic style
    dedent("""
    # INTEGRATED ARGUMENTATIVE ANALYSIS ASSIGNMENT

    OBJECTIVE: Conduct a comprehensive analysis of argumentative structure through complementary methods: textual annotation and standardized informal argument reconstruction.

    SOURCE MATERIAL:
    Analyze the following text:

    ::: {{.source_text}}
    {sources}
    :::

    ## PART I: TEXT ANNOTATION PROTOCOL

    TASK: Identify and annotate argumentative elements within the source text according to the specified schema.

    ANNOTATION SCHEMA:
    {annotation_scheme}

    METHODOLOGICAL REQUIREMENTS:
    1. Apply appropriate XML tags to demarcate propositions serving argumentative functions
    2. Assign unique identifiers to each proposition
    3. Document support and attack relationships between propositions
    4. Maintain textual integrity (permit abbreviation of non-argumentative segments only)
    5. Include reference attributes connecting to the argument reconstruction

    SUBMISSION FORMAT:
    Present your annotated text within a delimited code block with XML specification:
    ```xml
    // Annotated source text
    ```

    ## PART II: ARGUMENT RECONSTRUCTION PROTOCOL

    TASK: Produce an informal reconstruction of the primary argument(s) identified in the source text.

    METHODOLOGICAL REQUIREMENTS:
    1. Reconstruct a minimum of one (1) distinct argument in standard form
    2. For each argument:
       • Clearly label the argument
       • Identify all premises
       • Articulate intermediary conclusions as necessary
       • Formulate the final conclusion
    3. Document inferential pathways using standardized notation:
       • Format: `-- {{'from': ['premise_number', 'premise_number']}} --`
    4. Reference the corresponding text annotation via inline yaml data
       • Format: `(1) ... {{annotation_ids: ['proposition_id1', 'proposition_id2']}}`

    SUBMISSION FORMAT:
    Present your argument reconstructions within a delimited code block with Argdown specification:
    ```argdown
    // Standard-form argument reconstructions, separated by empty lines
    ```

    ## CROSS-METHODOLOGICAL COHERENCE REQUIREMENTS

    The two analytical components must demonstrate precise structural correspondence according to these parameters:

    1. Propositional Correspondence:
       • Each annotated proposition must reference its corresponding argument via `argument_label`
       • Each annotated proposition must reference its corresponding premise/conclusion via `ref_reco_label`
       • Each premise/conclusion must reference its textual sources via `annotation_ids` (possibly an empty list)
    
    2. Inferential Consistency:
       • Support relationships documented in the annotation must correspond to inferential relationships in the reconstruction
    
    EVALUATION CRITERIA:
    Your analysis will be assessed based on analytical precision, structural coherence, and adherence to cross-referential requirements.
           
    Submit your answer below, ensuring both components are enclosed in separate code blocks as detailed above!
    """).strip(),
    # Research-oriented style
    dedent("""
    # Dual Argument Analysis Protocol (ver. 3.2)

    RESEARCH CONTEXT:
    This protocol implements a bimodal analytical framework for argumentation, integrating textual annotation with standard-form reconstruction to enable comprehensive understanding of argumentative discourse.

    SOURCE DOCUMENT:
    ::: {{.source_text}}
    {sources}
    :::

    ## METHODOLOGY PHASE I: TEXTUAL ANNOTATION

    OBJECTIVE: Identify and document argumentative components within their original textual context.

    IMPLEMENTATION REQUIREMENTS:

    I. Annotation Schema Implementation
       {annotation_scheme}
    
    II. Propositional Identification Parameters
       A. Component Identification: Mark all textual segments serving argumentative functions
       B. Identifier Assignment: Provide unique ID for each argumentative segment
       C. Textual Integrity: Preserve original wording within annotated segments
       D. Relationship Documentation: Specify support/attack relations between components
       E. Cross-Reference Implementation: Include reference attributes connecting to reconstruction
    
    OUTPUT FORMAT:
    XML-encoded text enclosed in fenced code block (```xml ... ```)

    ## METHODOLOGY PHASE II: STANDARD-FORM RECONSTRUCTION

    OBJECTIVE: Develop Argdown representation of the arguments' internal inferential structure.

    IMPLEMENTATION REQUIREMENTS:
    
    I. Structural Components
       A. Argument Labeling: Clearly label each argument
       B. Premises: All supporting reasons
       C. Intermediate Conclusions: As necessary for complex arguments
       D. Final Conclusion: Ultimate proposition being established
    
    II. Inferential Documentation
       A. Format: `-- {{'from': ['premise_identifier', 'premise_identifier', ...]}} --`
       B. Comprehensiveness: Document all inferential pathways
    
    III. Cross-Reference Implementation
       A. Include reference attributes connecting to annotation
       B. Format: `(1) Premise text {{annotation_ids: ['proposition_id1', 'proposition_id2']}}`
    
    OUTPUT FORMAT:
    Argdown code enclosed in fenced code block (```argdown ... ```), with multiple arguments separated by newlines.

    ## CROSS-METHODOLOGICAL INTEGRATION REQUIREMENTS

    The analytical products must maintain precise correspondence through these mechanisms:

    I. Bidirectional Referencing System
       A. Annotation → Reconstruction:
          1. Each proposition must include argument_label attribute
          2. Each proposition must include ref_reco_label attribute
       B. Reconstruction → Annotation:
          1. Each premise/conclusion must include annotation_ids inline yaml data (possibly empty list)
    
    II. Structural Isomorphism
       Support relationships in annotation must correspond to inferential relationships in reconstruction
    
    This protocol ensures comprehensive analysis of argumentation through complementary representation systems with rigorous cross-referencing.
           
    Submit your analysis in a single answer below, including the two correctly formatted and fenced code blocks.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argument Analysis: Dual Representation Implementation

    ## Input
    ```
    Source text with argumentative content:
    {sources}
    ```

    ## Task Description
    Generate two linked representations of the source text's argumentative structure:
    1. XML annotation of source text
    2. Argdown premise-conclusion reconstruction(s)

    ## Implementation Requirements

    ### 1. Text Annotation Module
    
    Format: XML with custom schema
    Primary elements: <proposition> tags
    Required attributes:
      - id (unique identifier)
      - supports (optional, space-separated IDs)
      - attacks (optional, space-separated IDs)
      - argument_label (reference to Argdown argument)
      - ref_reco_label (reference to premise/conclusion in Argdown)
    

    Schema definition:
    ```
    {annotation_scheme}
    ```

    ### 2. Argument Reconstruction Module
    
    Format: Argdown argument in standard form
    Required elements:
      - Argument label (e.g., `<Argument Name>`)
      - Premises (numbered)
      - Conclusions (intermediate and final)
      - Inference paths with 'from' metadata
      - Inline yaml data for cross-referencing with annotation
    Minimum coverage: 1+ distinct arguments
    

    ### 3. Cross-Reference Implementation
    
    ```
    // Annotation → Reconstruction reference
    <proposition 
      id="p1" 
      argument_label="argument_name"
      ref_reco_label="1">
      Text content
    </proposition>    
    ```

    ```
    // Reconstruction → Annotation reference
    (1) Premise text {{annotation_ids: ['p1']}}
    ```    

    ### 4. Consistency Validation
    
    // Support relationship consistency
    For any propositions A,B in the annotation and corresponding statements A',B' in the Argdown reconstruction:
           A supports B => B' is inferred from A'

    ## Output Format
    1. XML annotation within fenced code block:
    ```xml
    <!-- Annotated text -->
    ```

    2. Argdown reconstruction within separate fenced code block:
    ```argdown
    // Argument reconstruction
    ```

    ## Development Notes
    - Maintain unchanged text content within annotations
    - Non-argumentative sections may be abbreviated
    - Final code blocks will be parsed programmatically
    - All cross-references must be bi-directional and consistent
           
    Carry out this task carefully and submit your results in a single answer below, ensuring both artifacts are enclosed in separate code blocks as specified.
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Analyzing Arguments: A Step-by-Step Guide

    In this exercise, you'll analyze an argument in two complementary ways: by annotating the source text and by reconstructing the argument in standard form. Let's break this down into manageable steps.

    ## The Text to Analyze

    First, carefully read this text:

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Identify the Key Components

    Before doing anything else, identify:
    - The main conclusion being argued for
    - The premises that support this conclusion
    - Any intermediate conclusions
    - How different parts of the argument relate to each other

    ## Step 2: Annotate the Source Text

    Now, add XML tags to mark the argumentative elements using this annotation scheme:

    {annotation_scheme}

    Follow these sub-steps:
    1. Identify each segment that functions as a proposition in the argument
    2. Wrap each segment in `<proposition>` tags
    3. Give each proposition a unique ID (e.g., "i1", "i2")
    4. Add the `supports` attribute to show which propositions support others
    5. Add the `argument_label` attribute to connect to your reconstruction
    6. Add the `ref_reco_label` attribute to specify which premise/conclusion it corresponds to

    When finished, place your annotated text in a code block:
    ```xml
    (your annotated text here)
    ```

    ## Step 3: Reconstruct the Argument

    Next, create a standard form reconstruction of the argument:

    1. List the premises with clear numbering
    2. Show how premises lead to intermediate conclusions (if any)
    3. Show how everything leads to the final conclusion
    4. For each conclusion, document which premises it follows from:
       `-- {{'from': ['1','2']}} --`
    5. Add `annotation_ids` metadata to each premise and conclusion:
       `(1) First premise {{annotation_ids: ['i1', 'i2']}}`

    Place your reconstruction in a separate code block:
    ```argdown
    (your argument reconstruction here)
    ```

    ## Step 4: Ensure Coherence Between Both Representations

    Finally, check that your annotation and reconstruction are properly connected:

    - Every proposition in the text has `argument_label` and `ref_reco_label` attributes
    - Every premise and conclusion has `annotation_ids` metadata
    - Support relationships in the annotation match inferential relationships in the reconstruction

    Double-check all IDs and labels to make sure they match correctly, and revise if necessary.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Creating Linked Annotations and Informal Argument Reconstructions

    In this tutorial, you'll learn how to analyze an argument through two complementary methods and link them together.

    ## The Text We'll Analyze

    Let's examine this text:

    ::: {{.source_text}}
    {sources}
    :::

    ## Method 1: Text Annotation

    The first step is to identify and annotate the argumentative components directly in the text.

    ### What is an Annotation?

    An annotation marks parts of the text that serve an argumentative function. We use XML tags to show:
    - Which segments function as propositions (claims, premises, or conclusions)
    - How these propositions relate to each other (support or attack)
    - How they connect to our argument reconstruction (which we'll create next)

    ### Annotation Schema:

    {annotation_scheme}

    ### How to Create Your Annotation:

    1. **Identify propositions** - Find sentences or phrases that make claims or provide reasons
    2. **Add tags** - Wrap each proposition in `<proposition>` tags
    3. **Assign IDs** - Give each proposition a unique identifier (e.g., "x1", "x2")
    4. **Mark relationships** - Show which propositions support or attack others
    5. **Add cross-references** - Include attributes that link to your reconstruction (may need to be adjusted later):
       - `argument_label` - Which argument this belongs to
       - `ref_reco_label` - Which premise or conclusion this corresponds to

    Put your completed annotation in a code block:
    ```xml
    (your annotated text here)
    ```

    ## Method 2: Argdown Argument Reconstruction

    Next, we'll create a standard form representation of the argument as premise-conclusion structure.

    ### What is a Standard Form Reconstruction?

    A standard form reconstruction shows the inferential structure of an argument by presenting:
    - Argument title (label)
    - Numbered premises (reasons)
    - Intermediate conclusions (if applicable)
    - The conclusion (what's being argued for)
    - The inferential relationships (how premises lead to conclusions)

    ### How to Create Your Reconstruction:

    1. **Create the argument title** - Give it a clear label (e.g., `<Main Argument>`)
    2. **List the premises** - Start with the supporting reasons
    3. **Add intermediate conclusions** - If needed for complex arguments
    4. **End with the main conclusion** - The ultimate point being argued for
    5. **Document inference paths** - Show which premises support each conclusion using:
       `-- {{'from': ['1','4']}} --`
    6. **Add cross-references** - Link back to your annotation using:
       `(1) First premise {{annotation_ids: ['x1', 'x2']}}`


    Put your completed reconstruction in a code block:
    ```argdown
    (your argument reconstruction here)
    ```

    ## Linking Both Methods Together

    The power of this approach comes from connecting both representations:

    - Every proposition in your annotation includes references to your reconstruction
    - Every premise and conclusion in your reconstruction references the corresponding text (possibly an empty list)
    - Support relationships in the annotation match inferential relationships in the reconstruction

    ### Example of Linked Components:

    In annotation:
    ```xml
    <proposition id="y1" argument_label="main_argument" ref_reco_label="1">All humans are mortal.</proposition>
    ```

    In reconstruction:
    ```argdown
    <Socrates>

    (1) All humans are mortal {{annotation_ids: ['y1']}}
    // ...
    ```

    Now try creating your own linked annotation and argument reconstruction for the text above!
    """).strip(),
]




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
        # randomly choose a prompt template
        self._prompt_template = random.choice(ARGANNO_PLUS_INFRECO_PROMPT_TEMPLATES)


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(sources=self.sources, annotation_scheme=ANNOTATION_SCHEME)

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
        prompt = "Revise your previously submitted annotation and argument reconstruction given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgannoPlusInfreco(Annotation, InformalReco):
    """
    Solution to the ArgannoPlusInfreco problem: annotation and argdown snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    def __str__(self):
        if self.annotated_source_text and self.argdown_snippet:
            return self.annotated_source_text + "\n\n" + self.argdown_snippet
        return self._raw_answer if self._raw_answer is not None else "None"
        
    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else str(self)

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgannoPlusInfreco":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)

        annotated_source_text = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml and vr.code_snippet
            ),
            None,
        )
        argdown_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )

        return cls(
            annotated_source_text=annotated_source_text,
            argdown_snippet=argdown_snippet,
            _raw_answer=raw_answer,
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


class ArgannoPlusInfrecoJudge(MPJudge):
    """Judge for the anno plus argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
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
        assert all(
            isinstance(solution, ArgannoPlusInfreco) for solution in solutions
        ), "All solutions must be ArgannoPlusInfreco objects"


    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgannoPlusInfrecoProblem), "Problem must be an ArgannoPlusInfrecoProblem"
        assert isinstance(solution, ArgannoPlusInfreco), "Solution must be an ArgannoPlusInfreco"

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler"),
                HasPCSHandler(name="InfReco.HasPCSHandler"),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler"),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler"),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler"),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler"),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler"),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler"),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler"),
            ]
        )

        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasAnnotationsHandler(),
                HasArgdownHandler(),
                ArgannoCompositeHandler(),
                infreco_handler,
                ArgannoInfrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation




class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid solutions
    where the source text's annotated propositions are textually similiar to the propositions in the reconstructed argument."""
    
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
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        soup = evaluation.artifacts["soup"]
        anno_props = soup.find_all("proposition")

        argdown = evaluation.artifacts["argdown_reco"]
        if argdown is None:
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

        #print("matches")
        #print(matches)
        dlss = [
            textdistance.damerau_levenshtein.normalized_similarity(s, t)
            for s, t in matches
        ]
        return round(sum(dlss) / len(dlss), 1) if dlss else 0.0
