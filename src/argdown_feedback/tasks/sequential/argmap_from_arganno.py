import random
from textwrap import dedent
from typing import Any, Sequence
from bs4 import BeautifulSoup
from pyargdown import ArgdownMultiDiGraph, Argument, Proposition, Valence
import textdistance

from argdown_feedback.tasks.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    GenericSolutionGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap
)
from argdown_feedback.tasks.core.arganno import (
    Annotation,
    AnnotationProblemGenerator,
    AnnotationJudge,
)


_ARGMAP_FROM_ARGANNO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    Assignment: Reconstruct a source text's argumentation as an argument map.
                        
    I will show you an annotated source text. The annotation identifies the argumentative function of different text segments. Your task is to reconstruct the text's arguments as an Argdown argument map.

    ::: {{.source_text}}
    {sources}
    :::

    In particular, I ask you

    - to explicitly label all nodes in your argument map;
    - to use square/angled brackets for labels to distinguish arguments/claims;
    - to indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
    - you may optionally refer, in your argument map, to text segments in the annotation through yaml inline data with an `annotation_ids` attribute that contains a list of proposition `ids`, being tolerant vis-à-vis potential errors in the annotation.
    
    DO NOT include any detailed reconstructions of individual arguments as premise-conclusion-structures in your argdown code.

    Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be map makers! 🗺️

    I have a special text that someone has already marked up with <proposition> tags. Your job is to turn it into a cool argument map!

    Here's the marked-up text:

    ::: {{.source_text}}
    {sources}
    :::

    Now, I want you to create an Argdown map that shows how all these marked parts fit together! Here's how:

    1. Look at all the <proposition> parts in the text and their IDs
    2. Create a map with all the main ideas (claims) and arguments
    3. For claims, use [square brackets] around their labels
    4. For arguments, use <angle brackets> around their labels
    5. Show which ideas support others with +> arrows
    6. Show which ideas attack others with -> arrows
    7. For each idea in your map, optionally tell us which marked parts it comes from using {{annotation_ids: ['id1', 'id2']}}
    8. Don't worry about any errors in the original marked-up text - just focus on making a clear map!

    When you're done, put your map between these special markers:
    ```argdown
    (your map goes here)
    ```

    Just focus on making a map that shows how everything connects - don't try to rebuild each argument with premises and conclusions.

    I can't wait to see your awesome map! 🌟
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! I've got an interesting task for you today.
                        
    I have this text that's already been annotated with proposition tags that mark the argumentative parts. What I need you to do is take that annotated text and turn it into an Argdown argument map.

    Here's the annotated text:

    ::: {{.source_text}}
    {sources}
    :::

    Basically, I need you to:
    - Create a representation of how all these tagged parts connect to each other (using Argdown syntax)
    - Use [square brackets] for claim labels and <angle brackets> for argument labels
    - Show which parts support or attack other parts
    - For each node in your map, you may include {{annotation_ids: ['id1', 'id2']}} to show which tagged parts in the original text it corresponds to

    Don't break down the individual arguments into premises and conclusions - just focus on the big picture of how everything relates.

    When you're done, just put your map in a code block starting with ```argdown and ending with ```.

    Thanks so much for your help with this!
    """).strip(),
    # Academic style
    dedent("""
    ASSIGNMENT: Construct an Argument Map from Annotated Source Text

    OBJECTIVE: Develop an Argdown argument map that represents the argumentative structure in the provided annotated text.

    ANNOTATED SOURCE MATERIAL:
    ::: {{.source_text}}
    {sources}
    :::

    REQUIREMENTS:

    1. STRUCTURAL REPRESENTATION
       • Extract all argumentative components identified in the source annotation
       • Represent these components as nodes in an Argdown argument map
       • Establish appropriate dialectical relations between nodes

    2. NOTATIONAL CONVENTIONS
       • Employ square bracket notation [label] for claims
       • Employ angle bracket notation <label> for arguments
       • Utilize standard Argdown syntax for support and attack relations
       • Maintain consistent labeling throughout the map

    3. CROSS-REFERENCE DOCUMENTATION (OPTIONAL)
       • For each node in your argument map, implement YAML inline data
       • Include an 'annotation_ids' attribute containing proposition IDs from the source text
       • Example format: {{annotation_ids: ['p1', 'p2']}}

    4. METHODOLOGICAL CONSTRAINTS
       • Omit detailed premise-conclusion reconstructions of individual arguments
       • Focus exclusively on dialectical relations between argumentative components
       • Adhere to standard Argdown syntactical conventions

    SUBMISSION FORMAT:
    Present your complete argument map within a single fenced code block demarcated with triple backticks and the argdown language identifier (```argdown) at the beginning and triple backticks (```) at the end.

    EVALUATION CRITERIA:
    Your argument map will be assessed based on comprehensiveness, structural accuracy, adherence to formatting requirements, and appropriate cross-referencing to source annotations.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Sequential Argument Mapping from Annotated Text
    
    PROCEDURAL CONTEXT:
    This protocol outlines the second phase of a sequential argumentative analysis, in which previously annotated text is transformed into a formal argument map structure.
    
    SOURCE MATERIAL (ANNOTATED TEXT):
    ::: {{.source_text}}
    {sources}
    :::
    
    ANALYTICAL OBJECTIVES:
    To derive a comprehensive visualization of the argumentative structure identified in the prior annotation phase.
    
    METHODOLOGICAL REQUIREMENTS:
    
    I. Source Analysis Parameters
       A. Extract all proposition elements from annotated text
       B. Identify dialectical relationships documented via supports/attacks attributes
       C. Preserve all ID references for cross-referential integrity
    
    II. Mapping Specifications
       A. Node Implementation
          1. Represent each argumentative component as a distinct node
          2. Implement typological differentiation:
             a. Claims: [square_bracket_notation]
             b. Arguments: <angled_bracket_notation>
          3. Add optional annotation referencing via YAML metadata:
             {{annotation_ids: ['source_id', 'source_id']}}
       
       B. Edge Implementation
          1. Represent all support relationships identified in annotation
          2. Represent all attack relationships identified in annotation
          3. Adhere to Argdown syntax conventions for dialectical relations
    
    III. Documentation Standards
       A. Maintain legal Argdown syntax throughout map
       B. Ensure comprehensive cross-referencing to source annotation
       C. Verify structural integrity of dialectical relationships
    
    IV. Representational Constraints
       A. Focus on macro-level argumentative structure rather than micro-level argument composition
       B. Omit detailed premise-conclusion reconstructions
    
    OUTPUT FORMAT:
    Present the completed argument map in a fenced code block with Argdown specification (```argdown ... ```)
    
    This protocol facilitates the transformation of linear text annotation into a structured visual representation of argumentative relationships.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argument Map Generation Task

    ## Input
    Annotated text with proposition markup:
    ```
    {sources}
    ```

    ## Task Description
    Transform XML-annotated text into an Argdown argument map that represents the argumentative structure.

    ## Input Format
    ```
    Type: XML with proposition annotation
    Key attributes: id, supports, attacks, argument_label
    ```

    ## Output Requirements

    ### Node Generation
    ```
    - Generate claim nodes for distinct statements
    - Generate argument nodes that support or attack other nodes
    - Use [square_brackets] for claim labels
    - Use <angle_brackets> for argument labels
    ```

    ### Relationship Mapping
    ```
    - Map all 'supports' attributes as support relations
    - Map all 'attacks' attributes as attack relations  
    - Use standard Argdown syntax:
      * Support: +> / <+
      * Attack: -> / <-
    - Be tolerant of potential errors in the annotation
    ```

    ### Cross-Reference Implementation (Optional)
    ```
    Format: {{annotation_ids: ['id1', 'id2']}}
    Purpose: Link map nodes to source text propositions
    ```

    ### Constraints
    ```
    - Do not include premise-conclusion structures
    - Focus on dialectical relationships between argumentative units
    - Ensure all proposition IDs from annotation are referenced
    ```

    ## Output Format
    Argdown map enclosed in fenced code block:
    ```argdown
    // argument map here
    ```

    ## Validation Criteria
    * All annotated propositions represented in map
    * All support/attack relationships preserved
    * Proper cross-referencing between map and annotation
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Creating an Argument Map from an Annotated Text: Step-by-Step Guide

    In this exercise, you'll transform an annotated text into an Argdown argument map. Let's break this down into simple steps:

    ## Step 1: Analyze the Annotated Text
    First, carefully examine the text with its proposition tags:

    ::: {{.source_text}}
    {sources}
    :::

    Pay special attention to:
    - The `id` attributes of each proposition
    - The `supports` and `attacks` attributes that show relationships
    - Any existing `argument_label` attributes

    ## Step 2: Identify the Components
    Make a list of:
    - All the main claims in the text
    - All the arguments that support or attack these claims
    - How these elements relate to each other

    ## Step 3: Create Your Map Structure
    Now, start building your map:
    1. Create nodes for each main claim using [square brackets]
    2. Create nodes for each argument using <angle brackets>
    3. Give each node a descriptive label

    ## Step 4: Establish Relationships
    Add the connections between nodes:
    - Use `+>` arrows to show support relationships
    - Use `->` arrows to show attack relationships

    ## Step 5 (Optional): Add Cross-References
    For each node in your map, add metadata showing which proposition(s) from the original text it corresponds to:
    ```
    {{annotation_ids: ['p1', 'p2']}}
    ```

    ## Step 6: Review Your Map
    Check that:
    - Every proposition from the annotated text is represented
    - All relationships match those in the annotation
    - Your map uses proper Argdown syntax
    - Any errors in the annotation are handled gracefully

    ## Step 7: Format Your Answer
    Put your completed map in a code block:
    ```argdown
    (your argument map here)
    ```

    Remember: Focus on the relationships between arguments and claims, not the internal structure of individual arguments.
    """).strip(),
    # Visualization-focused style
    dedent("""
    ARGUMENTATIVE STRUCTURE VISUALIZATION PROTOCOL

    OBJECTIVE: Transform annotated textual data into a visual representation of argumentative structure using Argdown notation.

    SOURCE CONTENT (ANNOTATED):
    ::: {{.source_text}}
    {sources}
    :::

    VISUALIZATION PARAMETERS:

    1. NODE EXTRACTION
       • Primary elements: Identify all <proposition> components from source annotation
       • Classification schema: 
         - Claims: Central positions being argued for/against
         - Arguments: Supporting or attacking reasoning units
       • Visual differentiation:
         - Claims: [square_bracket_notation]
         - Arguments: <angled_bracket_notation>
       • Reference system (optional!): {{annotation_ids: ['id1', 'id2']}} metadata

    2. EDGE REPRESENTATION
       • Directional indicators:
         - Support relationships: Extract from 'supports' attributes
         - Attack relationships: Extract from 'attacks' attributes
       • Visual syntax:
         - Support: +> / <+ notation
         - Attack: -> / <- notation

    3. METADATA INTEGRATION (OPTIONAL)
       • Cross-reference system: Every node linked to source propositions
       • Identification preservation: Maintain original IDs in annotation_ids attributes
       • Relationship fidelity: Dialectical relations in visualization must match source annotation

    OUTPUT FORMAT:
    Present visualization as structured Argdown code within fenced block:
    ```argdown
    // argument map here
    ```

    COMPLETION CRITERIA:
    The visualization must comprehensively represent all argumentative elements and their relationships while maintaining explicit references to the source annotation.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Creating an Argument Map from an Annotated Text

    In this tutorial, you'll learn how to transform a text that already has argumentative annotations into a clear argument map using Argdown notation.

    ## What You're Starting With

    You have a text where the argumentative parts have been marked with <proposition> tags:

    ::: {{.source_text}}
    {sources}
    :::

    ## What You'll Create

    You'll create an Argdown argument map that shows how all the identified propositions relate to each other in a visual structure.

    ## Why This Matters

    Converting from annotations to an argument map helps us:
    - See the big picture of how arguments connect
    - Visualize which claims are supported or attacked
    - Understand the overall argumentative structure
    - Uncover flaws in the annotation and handle them gracefully

    ## The Transformation Process

    ### 1. Understanding the Annotation
    
    First, look at what information the annotation provides:
    - Each `<proposition>` has a unique `id`
    - Some propositions `support` others
    - Some propositions `attack` others

    ### 2. Planning Your Map

    Based on these relationships:
    - Identify which propositions should be claims [in square brackets]
    - Identify which propositions should be arguments <in angle brackets>
    - Note which ones support or attack others

    ### 3. Creating Your Argument Map

    Now, build your map using Argdown syntax. Note that the yaml inline data is optional but helpful for cross-referencing:

    ```argdown
    // Claims with references to annotation IDs
    [claim1]: First major claim {{annotation_ids: ['p1']}}
    
    // Arguments with references to annotation IDs
    <arg1>: Supporting argument {{annotation_ids: ['p2']}}
    
    // Support and attack relationships
    <arg1>
        +> [claim1]
    ```

    ### 4. Adding Cross-References (Optional)

    For each node in your map, you may include the `annotation_ids` attribute that shows which proposition(s) in the original text it corresponds to.

    ## Your Task

    Now, create your own argument map based on the annotated text. Remember to:
    - Create nodes for all important propositions
    - Use the right bracket style for claims and arguments
    - Show all support and attack relationships
    - Put your map in a code block with ```argdown at the start and ``` at the end

    Good luck with your argument mapping!
    """).strip(),
]



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
        # randomly choose a prompt template
        self._prompt_template = random.choice(_ARGMAP_FROM_ARGANNO_PROMPT_TEMPLATES)


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(sources=self.annotated_text)

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

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
            soup_anno, _ = AnnotationJudge().parse_xml_snippet(arganno_solution[0].annotated_source_text)  # type: ignore
            return ArgmapFromArgannoProblem(
                annotated_text=str(arganno_solution[0]),
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
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgmapFromArgannoProblem)
        assert isinstance(solution, ArgumentMap)

        soup = problem.soup_anno
        if soup is None:
            return 0
        anno_props = soup.find_all("proposition")
        list_anno_props = "\n".join([ap.text for ap in anno_props])

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                list_anno_props, solution.argdown_snippet
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
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgmapFromArgannoProblem)
        assert isinstance(solution, ArgumentMap)

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

        #print("MATCHED_N", matched_n)

        return round(
            matched_n / (len(supports_anno) + len(attacks_anno)) if (len(supports_anno) + len(attacks_anno)) > 0 else 0,
            1,
        )

