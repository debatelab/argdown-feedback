import dataclasses
import random
from textwrap import dedent
from typing import Sequence

from pyargdown import ArgdownMultiDiGraph, Proposition

from argdown_feedback.tasks.base import (
    Evaluation,
    Feedback,
    MPJudge,
    Problem,
    ProblemGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.argmap import (
    ArgMapProblem,
    ArgumentMap,
    ConnectednessPreferencePairGenerator,
    MaxArgsPreferencePairGenerator,
    MaxAttacksPreferencePairGenerator,
    MaxSupportsPreferencePairGenerator,
    SourceTextProximityPreferencePairGenerator,
)
from argdown_feedback.tasks.core.infreco import InfRecoProblem, InformalReco
from argdown_feedback.verifiers.base import BaseHandler, CompositeHandler
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    ArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasArgdownHandler,
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
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)


_ARGMAP_PLUS_INFRECO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    # Assignment: Present a text's argumentation as an Argdown argument map, and informally reconstruct its arguments in standard form using Argdown syntax.
    
    Analyse the argumentation in the given **source text**. Your answer is supposed to contain two artifacts:
    1. an Argdown argument map and
    2. an Argdown snippet with informal reconstructions of all the arguments in standard form (premise-conclusion structure).

    In the following, you find
    * the source text to analyse,
    * detailed instructions for how to create the Argdown argument map (first artifact),
    * detailed instructions for how to reconstruct the arguments in standard form (second artifact),
    * a description of how both artifacts are supposed to cohere with each other,
    * formatting instructions for your answer.
           
    ## Source Text

    ::: {{.source_text}}
    {sources}
    :::

    ## Argument Mapping Task Details                   
           
    Create a syntactically correct Argdown argument map that comprehensively captures the macro argumentation in the text. In particular, you should

    - explicitly label all nodes in the argument map;
    - use square/angled brackets for labels to distinguish arguments/claims;
    - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

    Importantly, enclose your Argdown argument map in a fenced codeblock like so:
    ```argdown {{filename="map.ad"}}
    // your Argdown argument map here
    ```
    If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

    ## Argument Reconstruction Task Details                   

    Informally analyse and reconstruct the text's arguments with Argdown. In particular, you should

    - reconstruct the text's arguments in standard form (including premises, final 
      conclusion, and possible intermediate conclusions).
    - provide, for each conclusion in an argument, information about which previously introduced premises or 
      conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['1','4']}} --`,
      where the list items refer to the respective premise or conclusion labels.
    - ensure that every premise and intermediate conclusion is actually used to infer a conclusion in the argument.
          
    Importantly, enclose your Argdown reconstructions in a fenced codeblock:
    ```argdown {{filename="reconstructions.ad"}}
    // your Argdown argument reconstructions here
    ```
    If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.
           
    ## Required Coherence of Annotation and Argument Reconstructions                                            

    The argument map (first artifact) and your argument reconstructions (second artifact) must neatly correspond to each other. Meaning that:

    1. Every argument contained in the first argdown snippet ("map.ad") is reconstructed in standard form in the second argdown snippet ("reconstructions.ad"). So, for example, if your "map.ad" contains `<Argument A>`, `<Argument B>`, and `<Argument C>`, then these three arguments need to be informally reconstructed as premise conclusion structures in your "reconstructions.ad" snippet.
    2. Every argument reconstructed in the second argdown snippet ("reconstructions.ad") is present in the argument map. So, for example, if your "reconstructions.ad" also contains a reconstruction of `<Argument D>`, then `<Argument D>`, too, must figure in your "map.ad" code block.
    3. Whenever a claim in the argument map supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the reconstructed argument – and vice versa. So, for example, if your map contains the lines
    ```argdown {{filename="map.ad"}}
    // ...
    [Claim C]: Claim C.
        +> <Argument A>: ...
    // ...
    ```
    then the reconstruction of `<Argument A>` in your "reconstructions.ad" snippet must contain `[Claim C]` as premise.
    4. Whenever an argument in the argument map supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the reconstructed argument – and vice versa. So, for example, if your map contains the lines
    ```argdown {{filename="map.ad"}}
    // ...
    <Argument A>: Argument A.
        -> [Claim C]: Claim C.
    // ...
    ```
    then the reconstruction of `<Argument A>` in your "reconstructions.ad" snippet must feature "NOT: Claim C" as final conclusion.
    5. Whenever an argument A in the argument map supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) is a premise of B – and vice versa. So, for example, if your map contains the lines
    ```argdown {{filename="map.ad"}}
    // ...
    <Argument A>: Argument A.
        +> <Argument B>: Argument B.
    // ...
    ```
    then the reconstruction of `<Argument B>` in your "reconstructions.ad" snippet must contain A's conclusion as premise (or, if A attacks B, the negation of A's conclusion as premise).
    
    Here are the specific notation instructions which help you to ensure that your argument map and your argument reconstructions fully cohere with each other in the above sense: 

    - The argument labels in the argument map must match (1-to-1) the argument labels in the argument reconstruction.
    - Re-use the labels of claims in the argument map for the corresponding premises and conclusions (if any) in the argument reconstructions (premise-conclusion-structures). 
    - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label or, absent any label, have string-identical texts.
    - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if they have different labels, and one text prepends "NOT: " to the other text. (Avoid double negations and rely on duplex negatio affirmat instead.)
           
    Submit your answer below, including the Argdown argument map and the Argdown argument reconstructions in the required format. Make sure to use fenced code blocks for both artifacts, as described above.
    """).strip(),
    # Elementary school style
    dedent("""
    # Let's Be Argument Explorers! 🔍

    Hi there! Today we're going to be exploring arguments in a text and creating TWO special things:
    1. A cool argument map that shows how all the ideas connect
    2. A detailed breakdown of each argument showing all the reasons and conclusions

    ## The Text We're Exploring

    ::: {{.source_text}}
    {sources}
    :::

    ## Part 1: Create an Argument Map! 🗺️

    First, let's make a map that shows all the main arguments (at least two) and claims in the text and how they connect to each other.

    Here's how to make your map:
    1. Find all the main ideas (claims) and mark them with [square brackets]
    2. Find all the arguments and mark them with <angle brackets>
    3. Show which ideas support others with +> arrows (or <+ arrows)
    4. Show which ideas attack others with -> arrows (or <- arrows)
    5. Give every idea a clear label name           

    Put your finished map in a special box like this:
    ```argdown {{filename="map.ad"}}
    (your map goes here!)
    ```

    ## Part 2: Break Down the Arguments! 📋

    Now, let's take each argument from our map and show exactly how it works with premises (reasons) and conclusions:

    1. Take all the arguments from your map
    2. For each one, show:
       - All the reasons (premises) that support the conclusion
       - The main conclusion
       - Any middle steps (intermediate conclusions)
    3. Number each reason and conclusion
    4. For each conclusion, show which reasons it comes from using `-- {{'from': ['1','3']}} --`
    5. Make sure EVERY reason gets used in the argument!

    Put your argument breakdowns in another special box:
    ```argdown {{filename="reconstructions.ad"}}
    (your argument breakdowns go here!)
    ```

    ## SUPER IMPORTANT: Making Sure Everything Matches! 🔄

    The most important part is making sure your map and your argument breakdowns match perfectly:

    1. Arguments must match:
       - Every argument in your map needs to be broken down in Part 2
       - Every broken-down argument from Part 2 needs to be in your map
       - Keep map and breakdowns separate - don't break down any arguments in your map
    2. Support relationships must match:
       - If a claim supports an argument in your map, that same claim must be a reason in your breakdown
       - If an argument supports a claim in your map, that claim must be the conclusion in your breakdown
       - If one argument supports another in your map, the conclusion of the first must be a premise in the second
    3. Attack relationships must match:
       - If one claims attacks an argument your map, the negation of that claim must be in the breakdown
       - If an argument attacks a claim in your map, the negation of that claim must be the conclusion in your breakdown
       - If one argument attacks another in your map, the negation of the conclusion of the first must be a premise in the second

    Uff, that's a lot to keep track of! To help with this, use the SAME LABELS in both parts!

    To give you an idea of how it should look, here's a rough example of what your answer might look like:

    ```argdown {{filename="map.ad"}}
    [Main idea]: This is the main idea.
        <+ <Argument A>: This argument supports the main idea.
           <+ <Argument B>: This argument supports argument A.
    // add more arguments and claims as needed ...
    ```

    ```argdown {{filename="reconstructions.ad"}}
    <Argument A>
           
    (1) Reason 1 for Argument A.
    (2) Reason 2 for Argument A.
    -- {{'from': ['1', '2']}} --
    (3) [Main idea]: This is the main idea. // Argument A supports the main idea, that's why we put it here.

    // Now we add Argument B. 
    // Argument B supports Argument A, which means that one of A's premises is the conclusion of B.
    // But which premise? That can be tricky to decide! Let's assume Argument B backs Argument A's first premise:

    <Argument B>
           
    (1) Reason 1 for Argument B.
    (2) Reason 2 for Argument B.
    -- {{'from': ['1', '2']}} --
    (3) Reason 2 for Argument A. // Argument B supports Argument A, that's why we negate it here.
    ```

    I can't wait to see what great arguments you find! 🌟
    """).strip(),
    # Casual/friendly style
    dedent("""
    # Hey there! Let's map out some arguments together

    I've got this text I'd like you to analyze in two complementary ways - first by creating a visual map of how the arguments connect, and then by breaking down each argument into its premise-conclusion structure. It's kind of like looking at a city from above, and then exploring each building up close!

    ## Here's the text we're working with:

    ::: {{.source_text}}
    {sources}
    :::

    ## First task: Create an argument map

    I need you to create an Argdown argument map that shows the "big picture" of how all the arguments and claims in this text relate to each other. Include at least two arguments in your map. Think of it as drawing a network of ideas. For this map:

    - Label each claim with [square_brackets]
    - Label each argument with <angle_brackets>
    - Show when one thing supports another with "+>" / "<+" arrows
    - Show when one thing attacks or criticizes another with "->" / "<-" arrows

    When you're done, put your map in a code block like this:
    ```argdown {{filename="map.ad"}}
    // your map goes here
    ```

    ## Second task: Break down each argument

    Now I need you to "zoom in" on those arguments and show exactly how they work internally. For each argument:
    - Start every breakdown with the argument label in angle brackets (followed by an empty line)
    - Identify all the premises (the supporting reasons)
    - Show the main conclusion 
    - Include any intermediate conclusions
    - For each conclusion, show which premises it follows from using this format: `-- {{'from': ['2','3']}} --`
    - Make sure every premise actually gets used!
    - Prepend "NOT: " to indicate negations

    Put these breakdowns, separated by newlines, in another code block:
    ```argdown {{filename="reconstructions.ad"}}
    // your argument breakdowns go here
    ```

    ## Important: Keep everything connected!

    Here's the key part - make sure your map and your breakdowns match perfectly! Use the same labels in both parts, and make sure:
    - Every argument in your map gets broken down in detail
    - The support relationships in your map match what you show in your breakdowns, in line with Argdown conventions.
        * If a claim supports an argument in your map, it should be a premise in your breakdown
        * If an argument leads to a claim in your map, that claim should be the conclusion in your breakdown
        * If one argument supports another in your map, the first argument's conclusion should be a premise in the second
    - Likewise, the attack relationships in your map should match what you show in your breakdowns, in line with Argdown conventions.
        * If one claims attacks an argument your map, the negation of that claim must be in the breakdown
        * If an argument attacks a claim in your map, the negation of that claim must be the conclusion in the breakdown
        * If one argument attacks another in your map, the negation of the conclusion of the first must be a premise in the second

    ## Putting it all together
    
    Finally, make sure to format your answer with the two code blocks as described above. Roughly like so:
           
    ```argdown {{filename="map.ad"}}
    // your map goes here, for example:
    [Central claim]: First central claim.
        <+ <Supporting argument A>: Supporting argument for the first claim.
        <- <Counter argument B>: Counter argument against the first claim.
           <- <Another argument C>: Another argument against the counter argument.
    // more arguments and claims as needed ...
    ```

    ```argdown {{filename="reconstructions.ad"}}
    // your argument breakdowns go here, for example:
    <Supporting argument A>
           
    (1) Premise 1 of supporting argument.
    (2) Premise 2 supporting argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) [Central claim]: First central claim. // Argument A supports the central claim, that's why we put it here.
           
    <Counter argument B>

    (1) Premise 1 of counter argument.
    (2) Premise 2 of counter argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) NOT: First central claim. // Argument B attacks the central claim, that's why we negate it here.

    // We finally need to add argument C, which attacks argument B:
           
    <Another argument C>
           
    (1) Premise 1 of another argument.
    (2) Premise 2 of another argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) NOT: Premise 1 of counter argument. // Here we assume that Argument C attacks premise (1) of argument B.
    ```

    Thanks so much for helping with this! Looking forward to seeing how you map out these arguments.
    """).strip(),
    # Academic style
    dedent("""
    # COMPREHENSIVE ARGUMENT ANALYSIS ASSIGNMENT

    OBJECTIVE: Conduct a systematic analysis of argumentative structure using two complementary methodologies: macro-level argument mapping and micro-level informal argument reconstruction.

    SOURCE MATERIAL:
    Analyze the following text:

    ::: {{.source_text}}
    {sources}
    :::

    ## PART I: ARGUMENT MAPPING PROTOCOL

    TASK: Develop a comprehensive visual representation of the argumentative macrostructure present in the source text.

    METHODOLOGICAL REQUIREMENTS:
    1. Explicit node labeling for all argumentative components
    2. Typological differentiation between:
       • Claims: [square_bracket_notation]
       • Arguments: <angled_bracket_notation>
    3. Systematic documentation of dialectical relationships:
       • Support relations
       • Attack relations
    4. Minimum coverage of two (2) distinct arguments
    5. Adherence to standard Argdown syntactical conventions

    SUBMISSION FORMAT:
    Present your argument map within a delimited code block with filename specification:
    ```argdown {{filename="map.ad"}}
    // Argument map
    ```

    ## PART II: ARGUMENT RECONSTRUCTION PROTOCOL

    TASK: Produce informal reconstructions of multiple arguments in standard (premise-conclusion) form.

    METHODOLOGICAL REQUIREMENTS:
    1. Informally reconstruct distinct arguments as premise-conclusion structures
    2. For each argument:
       • Identify all premises
       • Articulate intermediary conclusions as necessary
       • Formulate the final conclusion
    3. Document inferential pathways using standardized notation:
       • Format: `-- {{'from': ['premise_number', 'premise_number']}} --`
    4. Ensure complete premise utilization (no superfluous premises)
    5. Use "NOT: " prefix for negated propositions

    SUBMISSION FORMAT:
    Present all your argument reconstructions, separated by newlines, within a single fenced code block with filename specification:
    ```argdown {{filename="reconstructions.ad"}}
    // Argument reconstructions
    ```

    ## STRUCTURAL COHERENCE REQUIREMENTS

    The two analytical components must demonstrate precise cross-referential integrity according to these parameters:

    1. Comprehensive Coverage:
       • Every argument in the map must have a corresponding reconstruction
       • Every reconstruction must have a corresponding map component
    
    2. Dialectical Consistency:
       • Support/attack relations in the map must be reflected in the reconstructions' premise-conclusion breakdowns
       • When claim C supports (attacks) argument A in the map, C (C's negation) must appear as a premise in A's reconstruction
       • When argument A supports (attacks) claim C in the map, C (C's negation) must appear as A's conclusion in the reconstruction
       • When argument A supports argument B in the map, A's conclusion (the negation of A's conclusion) must appear as a premise in B's reconstruction

    3. Notational Consistency:
       • Maintain identical labeling conventions across both analytical components
       • Use identical claim labels between map and reconstruction
       • Represent negations consistently using "NOT: " prefixing

    ## ABSTRACT EXAMPLE

    ```argdown {{filename="map.ad"}}
    // your map goes here, for example:
    [Central claim]: First central claim.
        <+ <Supporting argument A>: Supporting argument for the first claim.
        <- <Counter argument B>: Counter argument against the first claim.
           <+ <Another argument C>: Another argument against the counter argument.
    // more arguments and claims as needed ...
    ```

    ```argdown {{filename="reconstructions.ad"}}
    // your argument breakdowns go here, for example:
    <Supporting argument A>
           
    (1) Premise 1 of supporting argument.
    (2) Premise 2 supporting argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) [Central claim]: First central claim. // Argument A supports the central claim, that's why we put it here.
           
    <Counter argument B>

    (1) [Key premise]: Premise 1 of counter argument.
    (2) Premise 2 of counter argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) NOT: First central claim. // Argument B attacks the central claim, that's why we negate it here.

    // We finally need to add argument C, which supports argument B:
           
    <Another argument C>
           
    (1) Premise 1 of another argument.
    (2) Premise 2 of another argument.
    // ... more premises as needed ...
    -- {{'from': ['1', '2']}} --
    (3) [Key premise]: Premise 1 of counter argument. // Here we assume that Argument C backs premise (1) of argument B.
    ```
           

    EVALUATION CRITERIA:
    Your analysis will be assessed based on comprehensiveness, structural coherence, analytical precision, and adherence to formatting requirements.
    """).strip(),
    # Research-oriented style
    dedent("""
    # Multimodal Argumentative Analysis Protocol (v2.4)

    RESEARCH CONTEXT:
    This protocol implements a complementary analytical framework for argumentation, combining structural mapping with standard-form reconstruction methodologies to enable comprehensive understanding of argumentative discourse.

    SUBJECT TEXT:
    ::: {{.source_text}}
    {sources}
    :::

    ## METHODOLOGY PHASE I: MACROSTRUCTURAL MAPPING

    OBJECTIVE: Develop a comprehensive Argdown visualization of the argumentative structure.

    IMPLEMENTATION REQUIREMENTS:
    
    I. Node Specification Parameters
       A. Component Identification
          1. Claims: Propositions functioning as dialectical anchors
          2. Arguments: Reasoning units supporting or attacking claims
       B. Notational System
          1. Claims: [square_bracket_notation]
          2. Arguments: <angled_bracket_notation>
       C. Explicit Labeling System
          1. Unique identifiers for all components
          2. Consistency with Phase II reconstructions
    
    II. Relational Documentation
       A. Support Relations: Employ standard Argdown notation (+>/<+)
       B. Attack Relations: Employ standard Argdown notation (->/<-)
       C. Comprehensive Coverage: Document all significant dialectical relationships
    
    III. Size Parameters
       A. Minimum Argument Coverage: ≥2 distinct arguments

    OUTPUT FORMAT:
    Structured Argdown code enclosed in fenced block with metadata:
    ```argdown {{filename="map.ad"}}
    // Argdown argument map, no premise-conclusion structures
    ```

    ## METHODOLOGY PHASE II: MICROSTRUCTURAL RECONSTRUCTION

    OBJECTIVE: Develop informal reconstructions of individual arguments identified in Phase I in standard form.

    IMPLEMENTATION REQUIREMENTS:
        
    I. Structural Components
       A. Premises: All supporting reasons
       B. Intermediate Conclusions: As necessary for complex arguments
       C. Final Conclusion: Ultimate proposition being established
    
    II. Inferential Documentation
       A. Format: `-- {{'from': ['premise_identifier', 'premise_identifier']}} --`
       B. Comprehensiveness: Document all inferential pathways
       C. Premise Utilization: Ensure all premises contribute to conclusions
    
    OUTPUT FORMAT:
    Structured Argdown code enclosed in fenced block with metadata:
    ```argdown {{filename="reconstructions.ad"}}
    // Standard-form argument reconstructions, starting with argument labels in angle brackets and separated by newlines
    ```

    ## CROSS-METHODOLOGICAL COHERENCE REQUIREMENTS

    The two analytical products must maintain strict correspondence through the following mechanisms:

    I. Component Alignment
       A. Every argument in Phase I must have corresponding reconstruction in Phase II
       B. Every reconstruction in Phase II must have corresponding argument in Phase I
    
    II. Relational Consistency
       A. Map Support Relations → Reconstruction Premises/Conclusions
          1. If claim C supports argument A in map, then C appears as premise in A's reconstruction
          2. If argument A supports claim C in map, then C appears as conclusion in A's reconstruction
          3. If argument A supports argument B in map, then A's conclusion appears as premise in B's reconstruction
       B. Map Attack Relations → Reconstruction Negations
          1. If claim C attacks argument A in map, then the negation of C appears as premise in A's reconstruction
          2. If argument A attacks claim C in map, then the negation of C appears as conclusion in A's reconstruction
          3. If argument A attacks argument B in map, then the negation of A's conclusion appears as premise in B's reconstruction
       C. Negation Representation: Use "NOT: " prefix for negated propositions
    
    III. Label Coherence System
       A. Use identical labels across both analytical products
       B. Maintain consistent labeling between map nodes and reconstruction components

    This protocol ensures methodological triangulation and analytical robustness in the examination of argumentative discourse.
           
    Please submit your completed analysis of the SUBJECT TEXT with both required Argdown code blocks as specified above.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argument Analysis: Dual Format Implementation

    ## Input
    ```
    Source text with argumentative content:
    {sources}
    ```

    ## Task Description
    Generate two coordinated representations of argumentative structure:
    1. Macro-level argument map (dialectical structure)
    2. Micro-level argument reconstructions (inferential structure)

    ## Implementation Requirements

    ### 1. Argument Map Generation
    ```
    Type: Argdown map
    Node formats:
      - Claims: [square_bracket_labels]
      - Arguments: <angle_bracket_labels>
    Relations:
      - Support: +> or <+ 
      - Attack: -> or <-
    Output format: Fenced code block with {{filename="map.ad"}} metadata
    ```

    ### 2. Argument Reconstruction Implementation
    ```
    Type: Argdown premise-conclusion structures
    Format: Standard form arguments, starting with argument labels in angle brackets
    Required elements:
      - Premises (all must be used)
      - Conclusions (intermediate and final)
      - Inference paths via yaml inline data with `from` key
    Minimum coverage: 2+ distinct arguments
    Output format: Single fenced code block with {{filename="reconstructions.ad"}} metadata, containing multiple argument reconstructions separated by newlines
    ```

    ### 3. Cross-Format Coherence Module
    // Label consistency:
    - Map labels must match reconstruction labels        
           
    // Structural alignment
    - Argument <A> supports argument <B> in map iff, in the reconstruction, <A>'s conclusion is a premise in <B>
    - Argument <A> attacks argument <B> in map iff, in the reconstruction, <A>'s conclusion negates a premise in <B>
    - Claim [C] supports argument <A> in map iff, in the reconstruction, [C] is a premise in <A>
    - Claim [C] attacks argument <A> in map iff, in the reconstruction, [C] contradicts a premise of <A>
    - Argument <A> supports claim [C] in map iff, in the reconstruction, [C] is the conclusion of <A>
    - Argument <A> attacks claim [C] in map iff, in the reconstruction, the negation of [C] is the conclusion of <A>
           
    // Negation handling
    - Claim with string text "NOT: $P" is the negation of claim "$P"

    ## Output Format Specifications

    ### Argument Map
    ```argdown {{filename="map.ad"}}
    // Argument map implementation
    ```

    ### Informal Argument Reconstructions
    ```argdown {{filename="reconstructions.ad"}}
    // Argument reconstructions implementation  
    ```

    ## Validation Criteria
    * All map arguments have corresponding reconstructions
    * All reconstructions have corresponding map arguments
    * Support/attack relations in map align with premise/conclusion structures
    * Labels are consistent across both representations
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Creating Argument Maps and Reconstructions: Step-by-Step Guide

    In this assignment, you'll analyze arguments in a text using two complementary methods you're already familiar with. Let's break this down into manageable steps.

    ## The Text to Analyze

    First, carefully read this text:

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Identify the Key Components

    Before creating anything, identify:
    - The main claims being made
    - The arguments supporting or attacking those claims
    - How different arguments relate to each other

    ## Step 2: Create the Argdown Argument Map

    This map shows the "big picture" of how arguments and claims relate. It must contain at least 2 arguments.

    1. For each main claim in the text:
       - Create a node with [square_brackets]
       - Give it a clear, descriptive label

    2. For each argument in the text:
       - Create a node with <angle_brackets>
       - Give it a clear, descriptive label

    3. Add relationships:
       - Use +> arrows / <+ arrows to show support relationships
       - Use -> arrows / <- arrows to show attack relationships

    4. Review your map for completeness:
       - Are all major claims represented?
       - Are all significant arguments included?
       - Are all important relationships shown?
    
    5. Remove any informal argument reconstructions from the map – this is just for the macro structure!

    6. Format your map in a code block:
       ```argdown {{filename="map.ad"}}
       (your argument map here)
       ```

    ## Step 3: Create Argument Reconstructions

    Now, take the arguments from your map and break them down in detail.

    1. For each argument:
       - Identify all premises (supporting reasons)
       - Identify the main conclusion
       - Determine if there are any intermediate conclusions

    2. Format each argument:
       - Start with the argument label in <angle_brackets>, followed by an empty line
       - Number each premise and conclusion
       - For conclusions, show which premises they follow from using:
         `-- {{'from': ['1','2']}} --`
       - Make sure every premise is actually used

    3. Maintain coherence with your map:
       - Use the same labels as in your map
       - Ensure relationships match those in your map

    4. Format your reconstructions in a code block:
       ```argdown {{filename="reconstructions.ad"}}
       (your argument reconstructions here)
       ```

    ## Step 4: Ensure Coherence Between Map and Reconstructions

    Check that your map and reconstructions match perfectly:

    1. Compare arguments:
       - Every argument in the map should have a reconstruction
       - Every reconstructed argument should appear in the map

    2. Check support relationships:
       - If claim C supports argument A in the map, C should be a premise in A's reconstruction
       - If argument A supports claim C in the map, C should be A's conclusion in the reconstruction
       - If argument A supports argument B in the map, A's conclusion should be a premise in B

    3. Check attack relationships:
       - If claim C attacks argument A in the map, the negation of C should be a premise in A's reconstruction
       - If argument A attacks claim C in the map, the negation of C should be the conclusion in A's reconstruction
       - If argument A attacks argument B in the map, the negation of A's conclusion should be a premise in B

    4. Verify label consistency:
       - The same labels should be used in both the map and reconstructions

    ## Step 5: Review and Submit

    Before submitting:
    - Review both artifacts for completeness and accuracy
    - Ensure fenced code blocks are properly formatted
    - Verify that the coherence requirements are met

    Your final submission should contain both code blocks with their appropriate filename metadata.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Creating Argument Maps and Informal Reconstructions

    In this tutorial, you'll practice how to analyze arguments in two complementary ways: by creating an argument map to show the overall structure, and by reconstructing individual arguments to show their internal logic.

    ## The Source Text

    Let's start by reading the text we'll be analyzing:

    ::: {{.source_text}}
    {sources}
    :::

    ## What We'll Create

    We're going to create two different but connected views of the arguments in this text:

    1. **An argument map** - Shows the "big picture" of how claims and arguments relate
    2. **Argument reconstructions** - Shows the detailed structure of individual arguments

    ## Part 1: Creating an Argument Map

    An argument map visualizes the relationships between claims and arguments, showing what supports or attacks what.

    ### What to Include:
    - **Claims**: The main points being argued for or against (using [square brackets])
    - **Arguments**: Collections of reasons supporting or attacking claims (using <angle brackets>)
    - **Relationships**: Support relationships (+>, <+) and attack relationships (->, <-)

    ### Steps to Create Your Map:
    1. Identify the main claims in the text
    2. Identify the arguments that support or attack these claims
    3. Determine how these elements relate to each other
    4. Create your map using Argdown notation

    ### Stylized example:
    ```argdown
    [Climate Change]: Climate change is a serious threat.
        <+ <Scientific Consensus>: Scientific consensus supports climate change.
        <- <Economic Argument>: The economy can adapt.
    ```

    ### Format Your Map:
    Place your completed map in a code block with filename metadata:
    ```argdown {{filename="map.ad"}}
    // Your argument map here
    ```

    ## Part 2: Reconstructing Arguments

    Argument reconstruction breaks down individual arguments into premises and conclusions, showing the logical structure.

    ### What to Include:
    - **Premises**: The reasons or evidence supporting a conclusion
    - **Conclusions**: What follows from the premises
    - **Inference paths**: Which premises lead to which conclusions

    ### Steps to Reconstruct an Argument:
    1. Identify the premises and conclusion
    2. Number each premise and conclusion
    3. Add intermediate conclusions if necessary
    4. Show which premises support which conclusions
    5. Make sure every premise is used
    6. Use "NOT: " to indicate negations

    ### Example:
    ```argdown
    <Scientific Consensus>: Scientific consensus supports climate change.

    (1) 97% of climate scientists agree that climate change is a threat.
    (2) Scientific consensus is a reliable indicator of truth.
    -- {{'from': ['1','2']}} --
    (3) Climate change is a serious threat.

    <Economic Argument>: The economy can adapt.

    // ...
    ```

    ### Format Your Reconstructions:
    Place your completed reconstructions in a code block with filename metadata:
    ```argdown {{filename="reconstructions.ad"}}
    // Your argument reconstructions here
    ```

    ## Connecting Your Map and Reconstructions

    The power of this approach comes from connecting these two views. Here's how they should align:

    1. **Same Arguments**: Every argument in your map should be reconstructed, and every reconstruction should appear in your map
    2. **Same Support Relationships**: 
       - If claim C supports argument A in your map, C should be a premise in A's reconstruction
       - If argument A supports claim C in your map, C should be A's conclusion in its reconstruction
       - If argument A supports argument B in your map, A's conclusion should be a premise in B's reconstruction
    3. **Same Attack Relationships**:
       - If claim C attacks argument A in your map, the negation of C should be a premise in A's reconstruction
       - If argument A attacks claim C in your map, the negation of C should be the conclusion in A's reconstruction
       - If argument A attacks argument B in your map, the negation of A's conclusion should be a premise in B's reconstruction
    4. **Same Labels**: Use identical labels in both your map and reconstructions

    ## Your Task

    Now it's your turn! Create an argument map with at least two arguments from the text above and reconstruct the arguments in standard form. Make sure your map and reconstructions are coherent with each other.
           
    Take care to format your answer with the two fenced code blocks as described above.

    Good luck with your argument analysis!
    """).strip(),
]


class ArgmapPlusInfrecoProblem(InfRecoProblem, ArgMapProblem):
    """Task: Create coherent informal reco and argument map."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources
        # randomly choose a prompt template
        self._prompt_template = random.choice(_ARGMAP_PLUS_INFRECO_PROMPT_TEMPLATES)

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(sources=self.sources)

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
        prompt = "Revise your previously submitted argument map and argument reconstructions given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgmapPlusInfreco(Solution):
    """
    Solution to the ArgmapPlusInfreco problem: argmap and reconstructions snippet.

    Contains unparsed answer iff fenced code blocks couldn't be extracted.
    """

    argdown_map_snippet: str  | None = None
    argdown_reconstructions_snippet: str | None = None
    _raw_answer: str | None = None

    def __str__(self):
        if self.argdown_map_snippet and self.argdown_reconstructions_snippet:
            return self.argdown_map_snippet + "\n\n" + self.argdown_reconstructions_snippet
        return self._raw_answer if self._raw_answer is not None else "None"

    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else str(self)

    @classmethod
    def from_raw_answer(cls, raw_answer: str) -> "ArgmapPlusInfreco":
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)

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
            argdown_map_snippet=map_snippet,
            argdown_reconstructions_snippet=reco_snippet,
            _raw_answer=raw_answer,
        )

    def partial_argmap(self) -> ArgumentMap:
        """Return the argument map subsolution."""
        return ArgumentMap(
            argdown_snippet=self.argdown_map_snippet, _raw_answer=self._raw_answer
        )

    def partial_infreco(self) -> InformalReco:
        """Return the informal reconstruction subsolution."""
        return InformalReco(
            argdown_snippet=self.argdown_reconstructions_snippet, _raw_answer=self._raw_answer,
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


class ArgmapPlusInfrecoJudge(MPJudge):
    """Judge for the argmap plus infreco task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgmapPlusInfrecoProblem), (
            "Problem must be an ArgmapPlusInfrecoProblem"
        )
        assert (
            isinstance(original_solution, ArgmapPlusInfreco)
            or original_solution is None
        )
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgmapPlusInfreco) for solution in solutions
        ), "All solutions must be ArgmapPlusInfreco objects"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgmapPlusInfrecoProblem), "Problem must be an ArgmapPlusInfrecoProblem"
        assert isinstance(solution, ArgmapPlusInfreco), "Solution must be an ArgmapPlusInfreco"

        map_filter = BaseHandler.create_metadata_filter("filename", ["map.ad"])
        reco_filter = BaseHandler.create_metadata_filter(
            "filename", ["reconstructions.ad"]
        )

        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler", filter=reco_filter),
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
            ]
        )
        main_handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                ArgdownParser(name="ArgdownParser"),
                HasArgdownHandler(name="HasArgdownHandler.map", filter=map_filter),
                HasArgdownHandler(name="HasArgdownHandler.reco", filter=reco_filter),
                ArgMapCompositeHandler(filter=map_filter),
                infreco_handler,
                ArgmapInfrecoCoherenceHandler(),
            ]
        )
        request = VerificationRequest(inputs=str(solution), source=problem.sources)
        result = main_handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation



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
        solution: Solution,
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
            solution=solution.partial_argmap(),  # type: ignore
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
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
            solution=solution.partial_argmap(),  # type: ignore
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
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
            solution=solution.partial_argmap(),  # type: ignore
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
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
            solution=solution.partial_argmap(),  # type: ignore
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )


class SourceTextProximityPreferencePairGeneratorCT(
    SourceTextProximityPreferencePairGenerator
):
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
            solution=solution.partial_argmap(),  # type: ignore
            evaluation=Evaluation(
                is_valid=True, artifacts={"argdown_map": argdown}, metrics={}
            ),
        )
