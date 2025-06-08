import random
from typing import Sequence

import dataclasses
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import (
    ArgdownMultiDiGraph,
)
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
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap,
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
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.coherence.arganno_argmap_handler import (
    ArgannoArgmapCoherenceHandler,
)

_ARGMAP_PLUS_ARGANNO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    # Assignment: Annotate a source text, and reconstruct its argumentation as an Argdown argument map.
                
    Analyse the argumentation in a given **source text**. Your answer is supposed to contain two artifacts:
    1. an argumentative text annotation and
    2. an Argdown argument map.
           
    In the following, you find
    * the source text to analyse,
    * detailed instructions for how to annotate the source text (first artifact),
    * detailed instructions for how to create the Argdown argument map (second artifact),
    * a description of how both artifacts are supposed to cohere with each other,
    * formatting instructions for your answer.

    ## Source Text

    ::: {{.source_text}}
    {sources}
    :::

    ## Annotation Task Details                   
           
    Annotate the source text above according to the following schema:

    {annotation_scheme}

    Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                
    Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
           
    ## Argument Mapping Task Details                   

    Create a syntactically correct Argdown argument map that represents the overall argumentation in the text. In particular, you should

    - explicitly label all nodes in the argument map;
    - use square/angled brackets for labels to distinguish arguments/claims;
    - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

    Importantly, enclose your Argdown argument map in a separate fenced codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

    ## Required Coherence of Annotation and Argument Map

    The argument map and the annotated source text must cohere with each other. Every argument map node must correspond to an annotated text segment. Moreover, the support and attack relations in the argument map should reflect the annotated dialectical relations.
           
    In particular, you should ensure that: 

    - Every <proposition> element in the annotation has an `argument_label` attribute that refers to a node (label of claim or argument) in the argument map.
    - Every node in the Argdown argument map has yaml inline data with an `annotation_ids` attribute that contains a list of `id` attributes of the corresponding <proposition> elements in the annotation.
    - Two nodes in the argument map support each other if and only if the corresponding <proposition> elements are annotated to support each other (`support` attribute).
    - Two nodes in the argument map attack each other if and only if the corresponding <proposition> elements are annotated to attack each other (`support` attribute).
           
    ## Output Format
           
    Your answer must contain at least two fenced codeblocks: one for the annotated source text and one for the Argdown argument map. For example:
           
    ```xml
    // Annotated source text here
    ``` 
           
    ```argdown
    // Argdown argument map here
    ```
           
    Don't forget the three closing backticks for the fenced codeblocks!
    """).strip(),
    # Elementary school style
    dedent("""
    # Let's Be Argument Analysts! 🕵️‍♀️🔍

    Hello there! Today we're going to do TWO special activities with this text. First, we'll mark up the important parts, and then we'll make a cool map showing how all the ideas connect!

    ## The Text We're Going to Explore

    ::: {{.source_text}}
    {sources}
    :::

    ## Part 1: Finding and Marking the Arguments! 🖍️

    First, let's find all the important parts that make arguments in the text. We'll put these special tags around them:

    {annotation_scheme}

    Here's what to do:
    1. Find all the parts that make claims or arguments
    2. Put <proposition> tags around them
    3. Give each one a special ID number
    4. Show which parts support other parts
    5. Show which parts attack other parts
    6. Connect each part to our map (in Part 2)

    When you're done, put your marked-up text inside a special box:
    ```xml
    (your marked-up text goes here)
    ```

    ## Part 2: Making Our Argument Map! 🗺️

    Now let's make a map showing how all the ideas connect! For our map:
    
    1. Find all main ideas (claims) and put them in [square brackets]
    2. Find all arguments and put them in <angle brackets>
    3. Show which arguments support (+>/<+) or attack (->/<-) other ideas
    4. Give every idea a clear label

    Put your map in another special box:
    ```argdown
    (your argument map goes here)
    ```

    ## Important: Connecting Both Parts! 🔗

    Here's the SUPER IMPORTANT part - we need to connect our marked text to our map:

    - Every marked part in the text needs an `argument_label` that matches a label in your map
    - Every idea in your map needs to show which marked parts it comes from
    - If two parts support each other in the text, they must support each other in the map
    - If two parts attack each other in the text, they must attack each other in the map

    Remember to check that your two parts match perfectly!

    I can't wait to see what you discover! 🌟
    """).strip(),
    # Casual/friendly style
    dedent("""
    # Hey there! I need your help with a two-part analysis

    I'm working on this text and need to analyze its argumentative structure in two different ways. Could you help me out?

    ## The text I'm working with:

    ::: {{.source_text}}
    {sources}
    :::

    ## Part 1: Text annotation

    First, I need you to annotate the text using XML tags to mark all the argumentative parts. Here's the annotation scheme:

    {annotation_scheme}

    Basically, just add tags around the important parts that function as arguments. Don't change any wording - just add the tags and attributes. If there are long sections that aren't part of the argumentation, you can shorten those.

    When you're done with this part, put it in a code block that starts with ```xml and ends with ```.

    ## Part 2: Argument mapping

    Next, create an Argdown map showing how all these arguments connect. You'll need to:
    - Label each claim with [square brackets]
    - Label each argument with <angle brackets>
    - Show support and attack relationships between them

    Put this map in a separate code block starting with ```argdown and ending with ```.

    ## Super important: Making sure both parts match up

    Here's the critical part - your annotation and map need to match exactly:

    - Every annotated proposition needs an argument_label pointing to its corresponding node in the map
    - Every node in your map needs annotation_ids pointing to the corresponding text segments
    - Support and attack relationships must be consistent between both formats

    This way, I can see exactly which parts of the text correspond to which parts of the argument structure.

    Thanks so much for your help with this!
    """).strip(),
    # Academic style
    dedent("""
    # Argumentative Structure Analysis: Dual-Method Assignment

    OBJECTIVE: Analyze the argumentative structure of the provided text using two complementary methods: XML-based textual annotation and Argdown argument mapping.

    METHODOLOGY: This assignment requires the implementation of a two-phase analytical protocol with strict correspondence requirements between phases.

    ## Source Material

    Analyze the following text:

    ::: {{.source_text}}
    {sources}
    :::

    ## Phase I: Argumentative Annotation Protocol

    Apply the following annotation schema to identify the argumentative components within the source text:

    {annotation_scheme}

    Analytical Guidelines:
    * Mark up all propositions that serve an argumentative function
    * Assign unique identifiers to each proposition
    * Establish explicit support and attack relationships between propositions
    * Maintain textual integrity (modifications permitted only for non-argumentative segments)
    * Document correspondence between propositions and argument map elements

    Submission Format: Enclose the annotated text within a fenced code block with XML specification (```xml ... ```).

    ## Phase II: Argument Mapping Protocol

    Construct an Argdown argument map that formally represents the argumentative structure:

    Structural Requirements:
    * Utilize explicit node labeling for all argumentative elements
    * Employ conventional notation: square brackets for claims, angled brackets for arguments
    * Establish dialectical relations (support/attack) according to Argdown syntax conventions
    * Incorporate metadata linking map elements to textual propositions

    Submission Format: Enclose the argument map within a fenced code block with Argdown specification (```argdown ... ```).

    ## Coherence Requirements

    The dual representation system must maintain strict correspondence:

    * Each proposition element must reference its corresponding argument map node via the argument_label attribute
    * Each argument map node must reference its corresponding proposition element(s) via annotation_ids metadata
    * Support and attack relations must demonstrate bidirectional consistency between both representational systems
    * The collective argumentative structure must be internally consistent within and across representations
           
    Your submission must include both the annotated text and the argument map, formatted as specified above. Ensure that both representations are coherent and mutually consistent.

    EVALUATION CRITERIA: Analytical accuracy, structural coherence, adherence to notational conventions, and cross-representational consistency.
    """).strip(),
    # Research-oriented style
    dedent("""
    # Multimodal Argumentation Analysis Protocol Version 2.3

    RESEARCH OBJECTIVE: To develop a comprehensive representation of argumentative structure using bimodal annotation techniques.

    SOURCE DOCUMENT:
    ::: {{.source_text}}
    {sources}
    :::

    ANALYSIS METHODOLOGY

    This protocol implements a dual-representation framework for argumentative analysis, combining micro-level textual annotation with macro-level structural mapping. Both representations must maintain strict correspondence to ensure analytical validity.

    ## Protocol Component 1: Source Text Annotation

    Apply the following annotation schema to identify argumentative components within the source document:

    {annotation_scheme}

    Implementation Requirements:
    1. Identify all textual segments fulfilling argumentative functions
    2. Apply proposition tags with appropriate attribute documentation
    3. Establish explicit dialectical relationships (support/attack) between propositions
    4. Maintain referential integrity via unique identifiers
    5. Preserve original textual content (abbreviation permitted only for non-argumentative segments)

    Documentation Format: XML annotation enclosed in fenced code block (```xml ... ```)

    ## Protocol Component 2: Argument Structure Mapping

    Develop an Argdown representation of the argumentative macrostructure:

    Structural Parameters:
    1. Node identification with explicit labeling conventions
       - Claims: [square_bracket_notation]
       - Arguments: <angled_bracket_notation>
    2. Dialectical relation documentation using standardized symbolic representation
       - Support relations: <+ / +>
       - Attack relations: <- / ->
    3. Node-level metadata implementation with appropriate referencing

    Documentation Format: Argdown code enclosed in fenced code block (```argdown ... ```)

    ## Cross-Representational Coherence Requirements

    Establish bidirectional referential integrity between the two components:

    1. Textual Annotation → Argument Map:
       - Each proposition element must contain an argument_label attribute referencing its corresponding node
    
    2. Argument Map → Textual Annotation:
       - Each node must contain annotation_ids metadata referencing corresponding proposition elements
    
    3. Relational Consistency:
       - Support/attack relationships must demonstrate perfect correspondence between representations

    This protocol enables comprehensive analysis of argumentation at both textual and structural levels while maintaining systematic cross-referencing between representations.
           
    Submit your completed analysis of the above source text with both components formatted as specified above. Ensure that all elements are coherent and mutually consistent.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argument Structure Analysis: Dual-Format Implementation

    ## Input
    Text source containing argumentative content:
    ```
    {sources}
    ```

    ## Task Description
    Generate two linked representations of the text's argumentative structure:
    1. XML annotation of source text
    2. Argdown argument map

    ## Implementation Requirements

    ### 1. Source Text Annotation
    ```
    Format: XML with custom schema
    Primary elements: <proposition> tags
    Required attributes: id, argument_label
    Optional attributes: supports, attacks
    ```

    Schema definition:
    ```
    {annotation_scheme}
    ```

    ### 2. Argument Map Generation
    ```
    Format: Argdown syntax
    Node types: 
      - Claims [square_brackets]
      - Arguments <angle_brackets>
    Relationship types:
      - Support: <+ / +>
      - Attack: <- / ->
    Required metadata: annotation_ids
    ```

    ### 3. Cross-Reference Implementation
    ```
    // Annotation → Map reference
    <proposition id="p1" argument_label="claim1">Text content</proposition>
    ```

    ```
    // Map → Annotation reference
    [claim1]: Text content {{annotation_ids: ['p1']}}
    ```

    ### 4. Consistency Validation
    * All nodes in map must have corresponding text annotations
    * All annotated propositions must have corresponding map nodes
    * Support/attack relations must be identical in both representations

    ## Output Format
    1. XML annotation within fenced code block:
    ```xml
    <!-- Annotated text here -->
    ```

    2. Argdown map within separate fenced code block:
    ```argdown
    // Argument map here
    ```

    ## Development Notes
    - Maintain unchanged text content within annotations
    - Non-argumentative sections may be abbreviated
    - Final code blocks will be parsed programmatically for validation
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Argument Analysis Step-by-Step Guide

    In this exercise, you'll create two complementary representations of the argumentative structure in a text: an annotation and an argument map.

    ## The Text to Analyze

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Read and Understand the Text
    First, carefully read the text to identify:
    - The main claims being made
    - The arguments supporting or attacking those claims
    - How different parts of the text relate to each other

    ## Step 2: Create the Text Annotation
    Now, add XML tags to the text using this annotation scheme:

    {annotation_scheme}

    Follow these steps:
    1. Identify each segment that functions as a proposition in the argument
    2. Wrap each segment in `<proposition>` tags
    3. Give each proposition a unique ID (e.g., "p1", "p2")
    4. Identify which propositions support others using the "supports" attribute
    5. Identify which propositions attack others using the "attacks" attribute
    6. Add an "argument_label" attribute to connect each proposition to your argument map

    When finished, place your annotated text in a code block:
    ```xml
    (your annotated text here)
    ```

    ## Step 3: Create the Argument Map
    Next, create an Argdown map showing the structure of the argumentation:

    1. Create nodes for each claim using [square brackets]
    2. Create nodes for each argument using <angle brackets>
    3. Show support relationships with +> arrows
    4. Show attack relationships with -> arrows
    5. Add metadata to each node with {{annotation_ids: ['p1']}} to connect it to your annotation

    Place your argument map in a separate code block:
    ```argdown
    (your argument map here)
    ```

    ## Step 4: Ensure Coherence Between Both Representations
    Finally, check that your annotation and map match perfectly:

    - Every proposition in the text must have a corresponding node in the map
    - Every node in the map must have corresponding text in the annotation
    - Support relationships must be consistent across both representations
    - Attack relationships must be consistent across both representations

    Double-check all IDs and labels to make sure they match correctly!
    """).strip(),
    # Visualization-focused style
    dedent("""
    # Dual-Mode Argumentative Structure Visualization

    OBJECTIVE: Create a comprehensive visualization of argumentative structure using complementary representation formats.

    SOURCE CONTENT:
    ::: {{.source_text}}
    {sources}
    :::

    VISUALIZATION REQUIREMENTS:

    ## Primary Visualization: Text-Embedded Annotation
    
    MODE: XML markup with proposition tagging
    
    ANNOTATION SCHEMA:
    {annotation_scheme}
    
    VISUAL ELEMENTS:
    • Explicit boundary demarcation of argumentative units
    • Unique identifier assignment for each unit
    • Directional relationship indicators (support/attack)
    • Cross-reference linking to structural visualization
    
    OUTPUT FORMAT:
    ```xml
    <!-- Annotated text with embedded argumentative structure -->
    ```

    ## Secondary Visualization: Structural Argument Map
    
    MODE: Argdown notation with node-relationship mapping
    
    STRUCTURAL ELEMENTS:
    • Node representation with typological differentiation:
      - Claims: [square_bracket_notation]
      - Arguments: <angled_bracket_notation>
    • Edge representation with directional indicators:
      - Support relationships: +>/<+
      - Attack relationships: ->/<-
    • Cross-reference metadata linking to primary visualization
    
    OUTPUT FORMAT:
    ```argdown
    // Structural argument map
    ```

    ## Cross-Visualization Coherence Requirements
    
    REFERENTIAL INTEGRITY:
    1. Bidirectional linking between representations:
       • Text annotation → Map: `argument_label` attributes
       • Map → Text annotation: `annotation_ids` metadata
       
    2. Structural consistency across visualizations:
       • Identical support/attack relationship mapping
       • Comprehensive node-proposition correspondence
       • Consistent directional relationship orientation
    
    This dual-mode visualization approach enables both close textual analysis and holistic structural comprehension of the argumentative content.
           
    Make sure that both representations are coherent and submit your analyses as two separate code blocks, one for the annotated text and one for the argument map.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Creating Linked Annotation and Argument Maps

    In this tutorial, you'll learn how to analyze the argumentative structure of a text through two complementary methods and link them together.

    ## The Text We'll Analyze

    Let's examine this text:

    ::: {{.source_text}}
    {sources}
    :::

    ## Method 1: Text Annotation

    The first step is to identify and annotate the argumentative components directly in the text.

    ### What is an Annotation?

    An annotation marks parts of the text that serve an argumentative function. We use XML tags to show:
    - Which segments function as propositions (claims or premises)
    - How these propositions relate to each other (support or attack)
    - How they connect to our argument map (which we'll create next)

    ### Annotation Schema:

    {annotation_scheme}

    ### How to Create Your Annotation:

    1. **Identify propositions** - Find sentences or phrases that make claims or provide reasons
    2. **Add tags** - Wrap each proposition in `<proposition>` tags
    3. **Assign IDs** - Give each proposition a unique identifier (e.g., "1", "2")
    4. **Mark relationships** - Show which propositions support or attack others
    5. **Add map references** - Include `argument_label` attributes for linking to your map

    Put your completed annotation in a code block:
    ```xml
    (your annotated text here)
    ```

    ## Method 2: Argument Mapping

    Next, we'll create a visual representation of the argument structure.

    ### What is an Argument Map?

    An argument map shows the structure of reasoning using:
    - Nodes for claims and arguments
    - Arrows showing support and attack relationships

    ### How to Create Your Argument Map:

    1. **Create nodes** for claims using [square brackets]
    2. **Create nodes** for arguments using <angle brackets>
    3. **Show relationships** with arrows (+> or <+ for support, -> or <- for attack)
    4. **Add references** to your annotation using {{annotation_ids: ['1', '2']}}

    Put your completed map in a code block:
    ```argdown
    (your argument map here)
    ```

    ## Linking Both Methods Together

    The power of this approach comes from connecting both representations:

    - Every proposition in your annotation must reference its corresponding node in your map
    - Every node in your map must reference its corresponding proposition(s) in your annotation
    - Support and attack relationships must be consistent across both formats

    ### Example of Linked Components:

    In annotation:
    ```xml
    <proposition id="1" argument_label="claim1">The earth is round.</proposition>
    ```

    In map:
    ```argdown
    [claim1]: The earth is round {{annotation_ids: ['1']}}
    ```

    Now try creating your own linked annotation and argument map for the text above!
    """).strip(),
]


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
        # randomly choose a prompt template
        self._prompt_template = random.choice(_ARGMAP_PLUS_ARGANNO_PROMPT_TEMPLATES)

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
        prompt = "Revise your previously submitted annotation and argument map given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgmapPlusArganno(Annotation, ArgumentMap):
    """
    Solution to the ArgmapPlusArganno problem: annotation and argdown snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    annotated_source_text: str
    argdown_snippet: str
    _raw_answer: str

    def __str__(self):
        if self.annotated_source_text and self.argdown_snippet:
            return self.annotated_source_text + "\n\n" + self.argdown_snippet
        return self._raw_answer

    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else str(self)

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusArganno":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)

        annotated_source_text = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml and vr.code_snippet
            ),
            "",
        )
        argdown_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            "",
        )

        return cls(
            annotated_source_text=annotated_source_text,
            argdown_snippet=argdown_snippet,
            _raw_answer=raw_answer,
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


class ArgmapPlusArgannoJudge(MPJudge):
    """Judge for the anno plus argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgmapPlusArgannoProblem), (
            "Problem must be an ArgmapPlusArgannoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusArganno)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgmapPlusArganno) for solution in solutions
        ), "All solutions must be ArgmapPlusArganno objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgmapPlusArgannoProblem), "Problem must be an ArgmapPlusArgannoProblem"
        assert isinstance(solution, ArgmapPlusArganno), "Solution must be an ArgmapPlusArganno"

        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasAnnotationsHandler(),
                HasArgdownHandler(),
                ArgannoCompositeHandler(),
                ArgMapCompositeHandler(),
                ArgannoArgmapCoherenceHandler(),
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
    where the source text's annotated propositions are textually similiar to the node texts in the argument map."""

    hints = [
        "Make sure that your argument map stays faithful to and mimics closely "
        "the annotation of the source text. In particular, use a similar wording for claims as "
        "in the corresponding annotated source segments!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        soup = evaluation.artifacts.get("soup")
        argdown = evaluation.artifacts.get("argdown_map")
        assert soup and argdown, (
            "AnnotationProximityPreferencePairGenerator: Missing soup or argdown in evaluation artifacts"
        )
        assert isinstance(soup, BeautifulSoup), "soup must be a BeautifulSoup object"
        assert isinstance(argdown, ArgdownMultiDiGraph), (
            "argdown must be an ArgdownMultiDiGraph object"
        )

        dlss: list[float] = []
        for anno_prop in soup.find_all("proposition"):
            anno_label = anno_prop.get("argument_label")  # type: ignore
            anno_text = anno_prop.get_text()  # type: ignore
            ad_prop = next(
                (p for p in argdown.propositions if p.label == anno_label), None
            )
            if ad_prop and anno_text:
                for text in ad_prop.texts:
                    dlss.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text, anno_text
                        )
                    )
            ad_arg = next((a for a in argdown.arguments if a.label == anno_label), None)
            if ad_arg and anno_text:
                for text in ad_arg.gists:
                    dlss.append(
                        textdistance.damerau_levenshtein.normalized_similarity(
                            text, anno_text
                        )
                    )

        if not dlss:
            return 0.0

        return round(sum(dlss) / len(dlss), 1)
