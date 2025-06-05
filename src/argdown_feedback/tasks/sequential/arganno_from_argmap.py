import random
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


_ARGANNO_FROM_ARGMAP_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    Assignment: Apply a given annotation scheme to a source text.
                        
    I will show you a **source text** and an argument map. Your task is to annotate the **source text** in order to highlight the argumentative function of different parts in the text. The argument map is merely supposed to help you with this task.

    The source text to be annotated is:
           
    ::: {{.source_text}}
    {sources}
    :::

    The {qualifier}argument map which sketches the source text's argumentative structure:

    {argdown_snippet}

    Use the following schema to annotate the source text above:

    {annotation_scheme}

    In particular:           

    1. Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).                        
    2. Use the `argument_label` attribute to relate the annotated text segments to the given informal argument map.
    3. Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'.
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be text detectives! 🔍

    I have a story AND a map of the arguments in the story. Your job is to find and mark all the important parts in the story that match the map!

    First, here's the story we need to work on:

    ::: {{.source_text}}
    {sources}
    :::

    Now, here's the {qualifier}argument map that shows how all the ideas connect:

    {argdown_snippet}

    Your mission is to add special tags to the story that show where each part of the argument is! Use these special tags:

    {annotation_scheme}

    Here's how to complete your mission:

    1. Look at the argument map to see what claims and arguments are there
    2. Find those same ideas in the story
    3. Put <proposition> tags around each important idea you find 
    4. Give each one a special ID (like "1", "2", etc.)
    5. If one idea supports another, use the "supports" tag to show that!
    6. If one idea attacks another, use the "attacks" tag to show that!
    7. Optionally, you may add an "argument_label" for each part that matches what it's called in the map

    When you're done, put your work between these special markers:
    ```xml
    (your marked-up text goes here)
    ```

    And remember, don't change any of the words in the story - just add the tags around them!

    I know you can do this! Good luck, detective! 🌟
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! I need your help with something that's perfect for your skills.

    I've got this text and an argument map someone created for it. What I need you to do is basically "connect the dots" between them - annotate the original text to show where each part of the argument map comes from.

    Here's the text:

    ::: {{.source_text}}
    {sources}
    :::

    And here's the {qualifier}argument map that was created from it:

    {argdown_snippet}

    Could you add XML tags to the original text to show where each argument and claim from the map appears? Here's the annotation scheme to use:

    {annotation_scheme}

    Optionally, you may document that your annotation and the map align nicely: For each bit you tag with <proposition>, add an "argument_label" attribute that matches the label in the map (like [claim1] or <argument1>).

    Don't change any of the text itself - just add the tags. If there are super long sections that don't contain arguments, you can shorten those with [...].

    When you're done, just wrap everything in a code block starting with ```xml and ending with ```.

    This would be super helpful - thanks!
    """).strip(),
    # Academic style
    dedent("""
    TEXTUAL ANNOTATION EXERCISE: XML-Based Argument Markup with Reference Map

    OBJECTIVE: Apply a standardized annotation scheme to a source text using a provided argument map as reference.

    SOURCE MATERIALS:

    1. Primary Source Text:
    ::: {{.source_text}}
    {sources}
    :::

    2. Reference Argument Map ({qualifier}reconstruction of source text):
    {argdown_snippet}

    ANNOTATION PROTOCOL:

    Apply the following annotation schema to the source text, using the argument map as a guide for identifying argumentative components:

    {annotation_scheme}

    METHODOLOGICAL REQUIREMENTS:

    1. ANNOTATION STANDARDS
       • Enclose argumentative elements within <proposition> tags
       • Assign unique identifiers to each proposition element
       • Explicate support and attack relationships between propositions
       • Maintain textual integrity of the source (abbreviation permitted only for non-argumentative segments)
       
    2. CROSS-REFERENCE IMPLEMENTATION (OPTIONAL)
       • For each annotated proposition, include an argument_label attribute
       • Ensure all argument_label values correspond to nodes in the provided argument map
       • Maintain consistency between annotated dialectical relationships and map relationships
       • Be tolerant vis-à-vis mistakes in the argument map    
       
    3. SUBMISSION FORMAT
       • Enclose the annotated text within a fenced code block with XML specification
       • Begin code block with ```xml and terminate with ```
       • Ensure proper closure of all XML tags

    EVALUATION CRITERIA:
    This exercise assesses your ability to identify argumentative components in natural text guided by an argument map structure.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Argument Identification and Guided Annotation (Stage 2/2)
    
    PROCEDURAL CONTEXT:
    This protocol outlines a two-stage annotation procedure in which text segments are marked according to their argumentative function using a pre-established argument map as reference.
    
    SOURCE MATERIALS:
    
    Document A (Source Text for Annotation):
    ::: {{.source_text}}
    {sources}
    :::
    
    Document B ({qualifier}Reference Argument Map):
    {argdown_snippet}
    
    ANNOTATION SCHEMA:
    Apply the following annotation framework to Document A:
    {annotation_scheme}
    
    METHODOLOGICAL PROCEDURE:
    
    I. Text Segmentation Parameters
       A. Identify text segments corresponding to argumentative elements in Document B
       B. Demarcate segments using <proposition> tags
       C. Assign unique identifier attributes to each segment
       D. Preserve original text content (abbreviation permitted only for non-argumentative passages)
    
    II. Cross-Reference Heuristics (NOT REQUIRED, OPTIONAL)
       A. Establish explicit links between text segments and argument map elements
          1. Add argument_label attributes to each <proposition> element
          2. Ensure argument_label values correspond precisely to labels in Document B
       B. Document dialectical relationships between text segments
          1. Implement support relationships using "supports" attribute
          2. Implement attack relationships using "attacks" attribute
          3. Maintain consistency with relationships depicted in Document B
    
    III. Documentation Standards
       A. Maintain formal XML syntax throughout annotation
       B. Ensure proper attribute-value formatting
       C. Verify all references (IDs, argument_labels) are accurate
    
    OUTPUT FORMAT:
    Present completed annotation of Document A in a fenced code block with XML specification (```xml ... ```)
    
    This protocol ensures systematic identification of argumentative components. An explicit mapping between natural language expressions and formal argument representations is not required, but may be used as a heuristics.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Annotation Task Specification

    ## Input
    1. Source text document requiring argumentative markup:
    ```
    {sources}
    ```

    2. Reference argument map ({qualifier}Argdown):
    ```argdown
    {argdown_snippet}
    ```

    ## Task Description
    Implement XML-based argumentative annotation on source text using the argument map as reference structure.

    ## Annotation Schema
    {annotation_scheme}
           
    ## Implementation Requirements

    ### Required Elements
    * `<proposition>` tags around argumentative segments
    * Unique ID attributes for each proposition
    * `argument_label` attributes mapping to argument map nodes
    * Support/attack relationship attributes where applicable

    ### Relationships (Optional)
    // Map to Annotation
    [claim1] in map → <proposition argument_label="claim1"> in annotation
    <argument1> in map → <proposition argument_label="argument1"> in annotation
    
    // Dialectical Relationships
    A supports B in map → <proposition id="A" supports="B">
    A attacks B in map → <proposition id="A" attacks="B">
           
    ### Content Constraints           
    * Do not modify original text content
    * Non-argumentative segments may be abbreviated with [...]
    * Maintain structural integrity of all argumentative components

    ## Output Format
    Annotated text in XML format within fenced code block:
    ```xml
    <!-- annotated content here -->
    ```

    ## Completion Criteria
    * Content preserving markup of source text
    * Adherence to annotation schema
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Step-by-Step Guide: Annotating a Text Using an Argument Map

    In this exercise, you'll annotate a source text based on a pre-existing argument map. Let's break this down into manageable steps:

    ## Step 1: Understand Your Materials

    First, let's look at what you're working with:

    **Source Text** (this is what you'll annotate):
    ::: {{.source_text}}
    {sources}
    :::

    **{qualifier}Argument Map** (use this as your guide):
    {argdown_snippet}

    **Annotation Schema** (this defines how to mark up the text):
    {annotation_scheme}

    ## Step 2: Identify Key Components

    Look at the argument map and identify:
    - Each claim (in square brackets [claim])
    - Each argument (in angle brackets <argument>)
    - How they relate to each other (support/attack relationships)

    ## Step 3: Find Matching Text

    For each element in the argument map:
    1. Find the corresponding text in the source document
    2. Note where each claim and argument appears in the original text

    ## Step 4: Apply XML Tags

    For each piece of text you identified:
    1. Wrap it in `<proposition>` tags
    2. Add a unique `id` attribute (like "p1", "p2")
    3. Optionally, add an `argument_label` attribute that matches its label in the map
    4. If it supports other propositions, add a `supports` attribute with their IDs
    5. If it attacks other propositions, add an `attacks` attribute with their IDs

    ## Step 5: Heuristics

    Consider:
    - Does every node in the argument map have a corresponding tagged section in your annotation?
    - Do the support/attack relationships in your annotation match those in the map?
    - Are any flaws or inconsistencies in the argument map handled gracefully?

    ## Step 6: Format Your Answer

    Place your completed annotation in a code block:
    ```xml
    (your annotated text here)
    ```

    ## Example

    If the map has `[claim1]` and your text contains "The earth is round" that matches this claim, you would annotate it as:
    ```xml
    <proposition id="s1" argument_label="claim1">The earth is round</proposition>
    ```
    where the `argument_label` is optional.

    Now, begin your annotation by working through these steps systematically, and submit your correctly formatted annotation!
    """).strip(),
    # Visualization-focused style
    dedent("""
    DOCUMENT ANNOTATION SPECIFICATION: ARGUMENT-TEXT MAPPING PROJECT

    OBJECTIVE: Transform plain text into a structurally enriched document that explicitly marks argumentative components according to argument map representation.

    PRIMARY MATERIALS:

    1. SOURCE CONTENT (for annotation):
    ::: {{.source_text}}
    {sources}
    :::

    2. REFERENCE STRUCTURE ({qualifier}argument map):
    {argdown_snippet}

    ANNOTATION SCHEMA:
    {annotation_scheme}

    ANNOTATION PARAMETERS:

    1. STRUCTURAL MARKUP
       • Element demarcation: Enclose argumentative segments within <proposition> tags
       • Unique identification: Assign distinctive ID attributes to each element
       • Relationship indicators: Implement supports/attacks attributes to document dialectical relations
       • Optional grounding: Optionally include argument_label attributes to reference argument map nodes
    
    2. CONSTRAINTS
       • Content preservation: Maintain original textual content within annotated segments
    
    3. FORMATTING
       • Encapsulation: Present annotation within a fenced code block (```xml ... ```)
       • Syntax adherence: Maintain proper XML formation throughout
       • Element isolation: Ensure clear boundaries between annotated components (no nesting)
    
    This specification enables the creation of a multi-layered visualization that connects natural language argumentation to its structural representation.

    DELIVERABLE: XML-annotated text with optional mapping to the provided argument structure.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Creating an Annotated Text from an Argument Map

    In this tutorial, you'll learn how to annotate a text to show its argumentative structure by using an argument map as your guide.

    ## What You'll Need

    1. **Source Text** - The text we want to annotate:
       ::: {{.source_text}}
       {sources}
       :::

    2. **{qualifier}Argument Map** - A map showing the arguments in the text:
       {argdown_snippet}

    3. **Annotation Schema** - The rules for how to annotate:
       {annotation_scheme}

    ## What We're Trying to Accomplish

    Imagine the argument map as a "blueprint" of the text's argumentative structure. Your job is to go back to the "building" (the text) and label each room according to the blueprint.

    ## Step-by-Step Process

    ### 1. Understand the Argument Map
    First, look at the argument map and note:
    - The claims (marked with [square brackets])
    - The arguments (marked with <angle brackets>)
    - How they connect (which ones support or attack others)

    ### 2. Find the Matching Text
    For each element in the map, find where it appears in the original text. Look for sentences or paragraphs that express:
    - The exact claims from the map
    - The arguments described in the map

    ### 3. Apply the Annotation
    For each matching piece of text:
    - Wrap it in `<proposition>` tags
    - Add a unique `id` attribute (e.g., "S1", "S2")
    - If it supports other propositions, list their IDs in the `supports` attribute
    - If it attacks other propositions, list their IDs in the `attacks` attribute
    - Optionally add an `argument_label` attribute matching its label in the map, to keep track of your progress

    ### 4. Example
    If your map has:
    ```
    [claim1]: The earth is round
           
    <arg1>
        +> [claim1]
    ```

    You might annotate your text as:
    ```xml
    <proposition id="S1" argument_label="claim1">The earth is round.</proposition> <proposition id="S2" argument_label="arg1" supports="S1">Scientists have proven the earth is round.</proposition>
    ```

    ### 5. Final Format
    Make sure to put your entire annotation in a code block:
    ```xml
    (your annotated text here)
    ```

    Remember: Don't change the words in the text - just add the tags around them. If there are long non-argumentative sections, you can shorten them with [...].

    Now it's your turn to create your annotation!
    """).strip(),
]



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
        # randomly choose a prompt template
        self._prompt_template = random.choice(_ARGANNO_FROM_ARGMAP_PROMPT_TEMPLATES)


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        qualifier = "(arguably imperfect) " if self.argmap_evaluation and not self.argmap_evaluation.is_valid else ""
        prompt = self._prompt_template.format(
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
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        anno = solution
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
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        anno = solution
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
            matched_n / len(argdown_map.dialectical_relations) if len(argdown_map.dialectical_relations) > 0 else 0,
            1,
        )

