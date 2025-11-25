from textwrap import dedent

from models import ArgdownAnalysisTask


PROMPT_TEMPLATES = {
    "default": dedent("""\
        Please analyse the following argumentative text using Argdown:
        
        {source_text}
        """
    ),
    "arganno": dedent("""\
        Assignment: Apply a given annotation scheme to a source text.
                    
        Annotate the following **source text** in order to identify the argumentative function of different parts in the text.

        ::: {{.source_text word_count={word_count}}}
        {sources}
        :::

        Annotate the source text above according to the following schema:

        <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
        <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
        <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
        <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
        <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
        <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->

        Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts with more than 200 words may be shortened).

        Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'."""
    ),
    "argmap": dedent("""\
        Assignment: Reconstruct a source text's argumentation as an Argdown argument map.
                    
        Analyse the argumentation in the following source text by creating an Argdown argument map.

        ::: {{.source_text}}
        {sources}
        :::

        In particular, I ask you to

        - explicitly label all nodes in the argument map;
        - use square/angled brackets for labels to distinguish arguments/claims;
        - indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
        
        DO NOT include any detailed reconstructions of individual arguments as premise-conclusion-structures in your argdown code.

        Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'."""
    ),
    "infreco": dedent("""\
        Assignment: Reconstruct a source text's main argument in standard form.
                    
        Identify the main argument in the following source text and informally reconstruct it as premise-conclusion structure using Argdown.

        ::: {{.source_text}}
        {sources}
        :::

        Note in particular:

        - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
        ending with '```'. Just include a single Argdown codeblock in your answer.
        - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
        conclusion, and possible intermediate conclusions).
        - For each conclusion in the argument, provide information about which previously introduced premises or 
        intermediary conclusions it is inferred *from*: Use yaml inline data in the corresponding inference line right
        above the inferred conclusion, e.g. `-- {{'from': ['1','3']}} --`. The list items refer to the respective 
        premise or conclusion labels used in the inference step.
        - You may, but are in no way required to add additional information about which inference rules or argumentation
        schemes are applied in each sub-argument.
        - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
        summarize its gist in line with Argdown syntax conventions. 
            
        Carefully consider the following DON'Ts:

        - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument.
        - Do NOT add any inline dialectical relations in the premise conclusion structure.
        - Do NOT add any yaml inline data besides the required inference information.
        - Do NOT add any formalization of the argument's propositions (premises or conclusions) in your Argdown code."""
    ),
    "logreco": dedent("""\
        Assignment: Reconstruct a source text's main line of reasoning as a deductively valid argument in standard form.
                            
        Logically reconstruct the main argument in the following source text. Formalize all the premises and conclusions.
        Make sure the reconstructed argument is deductively valid and all premises are relevant.

        ::: {{.source_text}}
        {sources}
        :::

        Note in particular:

        - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
        ending with '```'. Just include a single Argdown codeblock in your answer.

        - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
        conclusion, and possible intermediate conclusions).

        - For each proposition in your reconstruction (premises and conclusions), provide an adequate propositional logic / FOL formalization in NLTK
        syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Minimal example:
        `(1) Socrates is mortal. {{formalization: 'F(a)', declarations: {{'a': 'Socrates', 'F': 'being mortal'}} }}`.
        Only declare variables that are used in the corresponding formalization and that have not been declared before.
        Ensure that your formalizations are consistent with each other.

        - For each inference step in the argument, provide information about which previously introduced premises or 
        conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`,
        where the list items refer to the respective premise or conclusion labels.
        
        - You may, but are in no way required to add additional information about which inference rules or argumentation
        schemes are applied in each sub-argument.

        - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
        summarize its gist in line with Argdown syntax conventions. 

        - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument."""
    ),
    "arganno_argmap": dedent("""\
        # Assignment: Annotate the source text, and reconstruct its argumentation as an Argdown argument map.
                    
        Analyse the argumentation in the given **source text**. Your answer is supposed to contain two artifacts:
        1. an argumentative text annotation and
        2. an Argdown argument map.
            
        In the following, you find
        * detailed instructions for how to annotate the source text (first artifact),
        * detailed instructions for how to create the Argdown argument map (second artifact),
        * a description of how both artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.

        ## Annotation Task Details                   
            
        Annotate the source text above according to the following schema:

        <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
        <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
        <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
        <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
        <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
        <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->

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
            
        Don't forget the three closing backticks for the fenced codeblocks!"""
    ),
    "arganno_infreco": dedent("""\
        # Assignment: Annotate the source text and informally reconstruct its main argument in standard form using Argdown syntax.
            
        Analyse the argumentation in the given **source text**. Your submission is supposed to contain two artifacts:
        1. an argumentative text annotation and
        2. an Argdown snippet with informal reconstructions of the main argumentation in standard form (premise-conclusion-structure).

        In the following, you find
        * detailed instructions for how to annotate the source text (first artifact),
        * detailed instructions for how to informally reconstruct the argumentation (second artifact),
        * a description of how both artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.
        

        ## Annotation Task Details                   
            
        Annotate the source text above according to the following schema:

        <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
        <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
        <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
        <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
        <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
        <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->

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
        ```"""
    ),
    "arganno_logreco": dedent("""\
        # Assignment: Annotate the source text and logically reconstruct its main argument in standard form using Argdown syntax.
        
        Analyse the argumentation in the given **source text**. Your submission is supposed to contain two artifacts:
        1. an argumentative text annotation and
        2. an Argdown snippet with logical reconstructions of the argumentation in standard form (as deductively valid inferences).

        In the following, you find
        * detailed instructions for how to annotate the source text (first artifact),
        * detailed instructions for how to logically reconstruct and formalize the main argumentation (second artifact),
        * a description of how both artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.
        
        ## Annotation Task Details           
                
        Annotate the source text above according to the following schema:

        <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
        <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
        <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
        <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
        <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
        <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->

        Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                    
        Enclose the annotated text in a fenced codeblock, starting with '```xml' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                
        ## Formal Argument Reconstruction Task Details                   

        Logically analyse and formally reconstruct the text's main argumentation as deductively valid argument(s) with Argdown.

        - Reconstruct *at least one argument* in standard form (including premises, final conclusion, and possible intermediate conclusions).                   
        - For each proposition in your reconstruction (premises and conclusions), provide an adequate propositional logic / FOL formalization in NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Only declare variables that are used in the corresponding formalization and that have not been declared before. Ensure that your formalizations are consistent with each other.
        - For each inference step in the argument(s), provide information about which previously introduced premises or conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.            
        - Provide a succinct label (title) for each argument and summarize its gist in line with Argdown syntax conventions. 
                
        Importantly, enclose all your formal reconstructions (separated by newlines) in a single fenced Argdown codeblock, starting with '```argdown' and ending with '```'. If you provide multiple argdown codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

                
        ## Required Coherence of Annotation and Formal Argument Reconstruction                                                

        The argument reconstruction, on the one hand, and the annotated source text, on the other hand, must cohere with each other. Moreover, the inferential relations in the reconstructed argument(s) should reflect the annotated support relations.
        In particular, you should ensure that:

        - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippet.
        - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding argument. 
        - Every premise and conclusion in an Argdown argument has yaml inline data. In addition to 'formalization' and 'declarations' keys, that yaml data has an `annotation_ids` attribute which contains a (possibly empty) list of `id` attributes of the corresponding <proposition> elements in the annotation.
        - If, in the annotation, one <proposition> element supports another one (via its `support` attribute), then, in the Argdown argument, the proposition corresponding to the former element is used to infer the conclusion corresponding to the latter element.

        Please encapsulate both artifacts in separate fenced codeblocks, for example:
                
        ```xml
        <!-- Annotated source text -->
        ```
                
        ```argdown
        // Argdown snippet with logical reconstructions
        ```"""
    ),
    "argmap_infreco": dedent("""\
        # Assignment: Present a text's argumentation as an Argdown argument map, and informally reconstruct its arguments in standard form using Argdown syntax.
        
        Analyse the argumentation in the given **source text**. Your answer is supposed to contain two artifacts:
        1. an Argdown argument map and
        2. an Argdown snippet with informal reconstructions of all the arguments in standard form (premise-conclusion structure).

        In the following, you find
        * detailed instructions for how to create the Argdown argument map (first artifact),
        * detailed instructions for how to reconstruct the arguments in standard form (second artifact),
        * a description of how both artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.
            
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
            
        Submit your answer below, including the Argdown argument map and the Argdown argument reconstructions in the required format. Make sure to use fenced code blocks for both artifacts, as described above."""
    ),
    "argmap_logreco": dedent("""\
        # Assignment: Present a text's argumentation as an informal Argdown argument map, and logically reconstruct its arguments in standard form using Argdown syntax.

        Analyse the argumentation in the given **source text**. Your answer is supposed to contain two artifacts:
        1. an Argdown argument map and
        2. an Argdown snippet with logical reconstructions of all the arguments in standard form (as deductively valid inferences).

        In the following, you find
        * detailed instructions for how to create the Argdown argument map (first artifact),
        * detailed instructions for how to logically reconstruct and formalize the arguments (second artifact),
        * a description of how both artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.
                        
        ## Argument Mapping Task Details                   
                
        Create a syntactically correct Argdown argument map that captures the overall argumentation in the text. In particular, you should

        - explicitly label all nodes in the argument map;
        - use square/angled brackets for labels to distinguish arguments/claims;
        - indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
        - cover *at least two* arguments in the argument map;

        Importantly, enclose your Argdown argument map in a fenced codeblock:
        ```argdown {{filename="map.ad"}}
        // your Argdown argument map here
        ```
        If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

        ## Argument Reconstruction Task Details                   

        Logically analyse and reconstruct the text's arguments with Argdown, ensuring the inferences are deductively valid.
        - Reconstruct all arguments presented in the map in standard form (including argument title, premises, final conclusion, and possible intermediate conclusions).      
        - For each premise and conclusion in your reconstructions, provide an adequate propositional logic / FOL formalization in NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Only declare variables that are used in the corresponding formalization and that have not been declared in the corresponding argument before. Ensure that your formalizations are consistent across different arguments.
        - For each inference step in the argument, provide information about which previously introduced premises or conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.
        - Use `<-` / `<+` / `><` syntax to declare that any premises and/or conclusions from different arguments logically entail or contradict each other, providing explicit labels for these claims in square brackets.

        Importantly, enclose your Argdown reconstructions in a single fenced codeblock, separating different arguments with newlines:
        ```argdown {{filename="reconstructions.ad"}}'
        // your formal Argdown reconstructions here
        ```
        If you provide multiple Argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

        ## Required Coherence of Annotation and Argument Reconstruction

        The argument map and your argument reconstructions must neatly correspond to each other. Meaning that:

        1. Every argument in the _argument map_ is reconstructed in standard form.
        2. Every reconstructed argument is present in the _argument map_.
        3. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) figures as premise in the reconstructed argument -- and vice versa.
        4. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) figures as conclusion in the reconstructed argument -- and vice versa.
        5. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then A's conclusion (or, respectively, its negation) figures as premise of B -- and vice versa.
        6. Whenever a claim A, in the _argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
                
        Here are the specific notation instructions which help you to ensure that your argument map, on the one hand, and your argument reconstructions, on the other hand, fully cohere with each other in the above sense: 

        - The argument labels in the argument map (angle brackets) must match (1-to-1) the argument labels in the argument reconstruction.
        - Re-use the labels of claims in the argument map (square brackets) for the corresponding premises and conclusions (if any) in the argument reconstruction.
        - In the argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
        - In the argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax)."""
    ),
    "arganno_argmap_logreco": dedent("""\
        # Assignment: Annotate the above source text, present its argumentation as an informal Argdown argument map, and logically reconstruct its arguments in standard form using Argdown syntax.

        Analyse the argumentation in the given **source text**. Your answer is supposed to contain three artifacts:
        1. an argumentative text annotation,
        2. an Argdown argument map, and
        3. logical reconstructions of all the arguments in standard form (as deductively valid inferences).

        In the following, you find
        * detailed instructions for how to annotate the source text (first artifact),
        * detailed instructions for how to create the Argdown argument map (second artifact),
        * detailed instructions for how to logically reconstruct and formalize the arguments in standard form (third artifact),
        * a description of how the three artifacts are supposed to cohere with each other,
        * formatting instructions for your answer.
        
        ## Annotation Task Details
                
        Annotate the source text above according to the following schema:

        <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
        <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
        <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
        <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
        <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
        <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->

        Add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts may be shortened).
                    
        Enclose the annotated text in a fenced codeblock, starting with '```xml {{filename="annotation.txt"}}' and ending with '```'. If you provide multiple xml-codeblocks (e.g., improved versions or revisions), we will use and evaluate the last one only.
                                    
        ## Argument Mapping Task Details

        Create a syntactically correct Argdown argument map that captures the argumentation in the text (with at least two arguments). In particular, you should

        - explicitly label all nodes in the argument map;
        - use square/angled brackets for labels to distinguish arguments/claims;
        - indicate support and attack relations between nodes in accordance with Argdown syntax conventions.

        Importantly, enclose your Argdown argument map in a fenced codeblock, starting with '```argdown {{filename="map.ad"}}' and ending with '```'. If you provide multiple argdown map codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

        ## Logical Argument Reconstruction Task Details                   

        Logically analyse and formally reconstruct the text's arguments with Argdown, ensuring the inferences are deductively valid.

        - Reconstruct all arguments in standard form (including premises, final conclusion, and possible intermediate conclusions).                   
        - For each proposition in your reconstruction (premises and conclusions), provide an adequate propositional logic / FOL formalization in NLTK syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Only declare variables that are used in the corresponding formalization and that have not been declared in the corresponding argument before. Ensure that your formalizations are consistent across different arguments.
        - For each inference step in the argument, provide information about which previously introduced premises or conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['1','3']}} --`, where the list items refer to the respective premise or conclusion labels.
        - Use `<-` / `<+` / `><` syntax to declare that any premises and/or conclusions from different arguments logically entail or contradict each other, providing explicit labels for these claims in square brackets.
        - Start each logical reconstruction with the <argument title> and an optional short description of the argument's gist. 
                
        Importantly, enclose your all your logical reconstructions in a single fenced Argdown codeblock (separated by empty lines), starting with '```argdown {{filename="reconstructions.ad"}}' and ending with '```'. If you provide multiple argdown reconstructions codeblocks (e.g., improved versions or revisions), we will use and evaluate the last of these only.

        ## Required Coherence of Annotation, Argument Map, and Argument Reconstructions

        The annotation, the argument map and your argument reconstructions must neatly correspond to each other. Meaning that:

        The argument reconstructions and the annotated source text must cohere with each other. Moreover, the inferential relations in the logically reconstructed arguments must reflect the annotated support relations. That is:
        
        1. Every argument in the _argument map_ is logically reconstructed in the _Argdown reconstructions_.
        2. Every reconstructed argument in the _Argdown reconstructions_ is present in the _argument map_.
        3. Every annotated text segment in the _annotation_ corresponds to a premise or conclusion in a reconstructed argument in the _Argdown reconstructions_.
        4. Whenever a claim in the _argument map_ supports (attacks) an argument, the corresponding claim (or, respectively, its negation) is a premise in the corresponding reconstructed argument -- and vice versa -- in the _Argdown reconstructions_.
        5. Whenever an argument in the _argument map_ supports (attacks) a claim, the corresponding claim (or, respectively,  its negation) is the conclusion in the corresponding reconstructed argument -- and vice versa -- in the _Argdown reconstructions_.
        6. Whenever an argument A in the _argument map_ supports (attacks) another argument B, then, in the _Argdown reconstructions_, A's conclusion (or, respectively, its negation) is a premise of B -- and vice versa -- in the _Argdown reconstructions_.
        7. Whenever a claim A, in the _Argdown reconstructions_, is declared to support, attack, or contradict another claim B, then the formalizations of A and B must logically ground this relation.
        8. Whenever a text segment A in the _annotation_ supports another text segment B, then, in the _Argdown reconstructions_, B's corresponding proposition is inferred from the proposition corresponding to A, or A refers to an argument that supports the argument referenced by B.
        9. Whenever a text segment A in the _annotation_ attacks another text segment B, then, in the _Argdown reconstructions_, A's corresponding argument attacks the argument referenced by B.
        
        Here are the specific notation instructions which help you to ensure that annotation (filename="annotation.txt"), argument map (filename="map.ad") and argument reconstructions (filename="reconstructions.ad") fully cohere with each other in the above sense: 

        - Every <proposition> element in the annotation has an `argument_label` attribute, which refers to a label of an argument in the Argdown snippets.
        - Every <proposition> element in the annotation has a `ref_reco_label` attribute, which refers to a label of a premise or conclusion in the corresponding logically reconstructed argument (_Argdown reconstructions_).
        - Every premise and conclusion in the Argdown argument reconstructions has yaml inline data. Besides 'formalization' and 'declarations' keys, that yaml data has an `annotation_ids` attribute which contains a (possibly empty) list of `id` attributes of the corresponding <proposition> elements in the annotation.
        - The argument labels in the argument map match (1-to-1) the argument labels in the argument reconstruction.
        - The labels of claims in the argument map are re-used for the corresponding premises and conclusions (if any) in the argument reconstruction. 
        - In the Argdown argument reconstructions, two propositions (premise or conclusion) count as the same if they have the same label.
        - In the Argdown argument reconstructions, one proposition (premise or conclusion) counts as the negation of another proposition (premise or conclusion) if a corresponding logical relation between them is defined in the argdown snippet (e.g., with "><" or "->" syntax).
        
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
        ```"""
    ),
}


def get_base_instruction(
    task_id: ArgdownAnalysisTask,
    subtask_id: str | None,
    source_text: str,
) -> str:
    """Generate the base instruction for the given task and subtask."""

    kwargs: dict = {}
    if subtask_id in ["arganno"]:
        word_count = len(source_text.split())
        kwargs["word_count"] = word_count
    if subtask_id in ["arganno", "argmap", "infreco", "logreco"]:
        kwargs["source_text"] = source_text

    if subtask_id not in PROMPT_TEMPLATES:
        subtask_id = "default"
    return PROMPT_TEMPLATES[subtask_id].format(**kwargs)