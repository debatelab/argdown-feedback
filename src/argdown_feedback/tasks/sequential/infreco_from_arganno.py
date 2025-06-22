import random
from textwrap import dedent
import textdistance

from argdown_feedback.tasks.base import (
    Problem,
    Evaluation,
    ProblemGeneratorLLM,
    GenericSolutionGenerator,
    ScoringVirtuePreferencePairGenerator,
    Solution,
)
from argdown_feedback.tasks.core.infreco import (
    InfRecoProblem,
    InformalReco
)
from argdown_feedback.tasks.core.arganno import (
    Annotation,
    AnnotationProblemGenerator,
    AnnotationJudge,
)


_INFRECO_FROM_ARGANNO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    Assignment: Reconstruct a source text's main argument in standard form.
                        
    Reconstruct the argument presented in the following annotated source text in Argdown standard form (as premise-conclusion-structure). Use the argumentative markup to identify the main argument and to guide your informal reconstruction.

    ::: {{.source_text}}
    {sources}
    :::

    Note in particular:

    - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
      ending with '```'. Just include a single Argdown codeblock in your answer.
    - In your Argdown snippet, only reconstruct *a single argument* in standard form (including premises, final 
      conclusion, and possible intermediate conclusions), no matter whether the annotation highlights more than
      one argument.
    - For each conclusion in the argument, provide information about which previously introduced premises or 
      conclusions it is inferred *from*, using yaml inline data in the inference line, e.g. `-- {{'from': ['2','5']}} --`,
      where the list items refer to the respective premise or conclusion labels.
    - You may, but are in no way required to add additional information about which inference rules or argumentation
      schemes are applied in each sub-argument.
    - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
      summarize its gist in line with Argdown syntax conventions. 
           
    Carefully consider the following DON'Ts:

    - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument.
    - Do NOT add any inline dialectical relations in the premise conclusion structure.
    - Do NOT add any yaml inline data besides the required inference information.
    - Do NOT add any formalization of the argument's propositions (premises or conclusions) in your Argdown code.
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be argument builders! 🏗️

    I want you to look at the text you already marked up with <proposition> tags, and now turn it into a clear argument!

    Here's the text with the annotations you made:

    ::: {{.source_text}}
    {sources}
    :::

    Your mission is to create ONE neat argument using the parts you already found and marked. Look especially for:
    - Which parts were marked as "supports" (these might be your premises)
    - Which parts were referred to as "supported" (these might be your conclusion)
    - How the different parts connect together

    Here's how to complete your mission:

    1. First, find the main conclusion (what the whole argument is trying to prove)
    2. Then find all the premises that support it
    3. Show how the premises connect to make the conclusion
    4. Note that you might have to add premises or conclusions that are not directly in the text

    When you write down the argument:
    - Put everything in a special code box that starts with ```argdown and ends with ```
    - Give your argument a cool title at the top
    - Number each reason (premise)
    - For each conclusion, show which reasons it comes from using this special code: `-- {{'from': ['1','3']}} --` (this means the conclusion comes from reasons 1 and 3)
    - Only show ONE main argument (not lots of different arguments)

    Important things NOT to do:
    - Don't include any extra arguments or maps
    - Don't add any special symbols like + or -> between premises
    - Don't add any extra code besides the 'from' information
    - Don't add any fancy logic symbols or equations

    I know you can do this! Good luck, argument builder!
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! Let's take that annotation you created and turn it into a clear argument structure.

    Here's the annotated text you're working with:

    ::: {{.source_text}}
    {sources}
    :::

    So we've already got the groundwork done with those annotations you created. Now I'd like you to extract the main argument and put it in a nice, clean standard form. Here's what I'm looking for:

    - Create ONE main argument structure showing how premises lead to a conclusion
    - Format everything in Argdown syntax inside a code block (```argdown ... ```)
    - Make sure to include a clear title at the top
    - For each conclusion, show which premises it builds from with `-- {{'from': ['1','4']}} --` notation
    
    A few things to avoid:
    - No need for multiple arguments - just focus on the main one
    - Skip any support/attack indicators between statements
    - Keep it simple - just the numbered statements and inference connections
    - No formalization or symbolic logic needed

    Basically, I want to see how those annotated propositions fit together into a coherent argument flow. Thanks!
    """).strip(),
    # Academic style
    dedent("""
    ASSIGNMENT: Argument Reconstruction Based on Prior Annotation
    
    OBJECTIVE: Construct a formal premise-conclusion structure derived from your previous argumentative annotation of the source text.
    
    ANNOTATED SOURCE:
    ::: {{.source_text}}
    {sources}
    :::
    
    RECONSTRUCTION PARAMETERS:
    
    1. STRUCTURAL REQUIREMENTS
       • Identify a single coherent argument from the annotated propositions
       • Arrange the argument in standard form with premises preceding conclusions
       • Ensure meaningful progression from premises to final conclusion, possibly via intermediary conclusions
       • Make implicit premises or conclusions explicit as necessary
    
    2. INFERENTIAL DOCUMENTATION
       • For each conclusion (intermediate or final), document its inferential basis
       • Implement standardized notation: `-- {{'from': ['premise_numbers']}} --`
       • Ensure numerical references correspond to your reconstruction's labeling system
       
    3. FORMATTING SPECIFICATIONS
       • Begin with a concise, descriptive title encapsulating the argument
       • Enclose the entire reconstruction within a delimited code block (```argdown ... ```)
       • Number all components sequentially for clear reference
       
    4. METHODOLOGICAL CONSTRAINTS
       • Limit reconstruction to a single argument structure
       • Omit dialectical relations within the premise-conclusion structure
       • Exclude extraneous yaml metadata beyond required inference documentation
       • Refrain from formalizing propositions with logical notation
    
    EVALUATION CRITERIA:
    Your reconstruction will be assessed based on alignment with your previous annotation, structural coherence, inferential clarity, and adherence to formatting requirements.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Sequential Argument Analysis Phase II - Structural Reconstruction
    
    PROCEDURAL CONTEXT:
    This protocol outlines the second phase of a two-stage argumentative analysis, building upon the XML annotation completed in Phase I.
    
    SOURCE MATERIAL (ANNOTATED TEXT):
    ::: {{.source_text}}
    {sources}
    :::
    
    ANALYTICAL OBJECTIVES:
    To derive a structured representation of the principal line of argumentation identified in the prior annotation phase.

    METHODOLOGICAL REQUIREMENTS:
    
    I. Source Analysis Parameters
       A. Utilize the proposition elements and their relations as marked in the XML annotation
       B. Identify the central argumentative thread for reconstruction
       C. Note particularly the support relations indicated in the annotation
    
    II. Reconstruction Specifications
       A. Extract a single, coherent argument from the annotated material
       B. Arrange components in standard form
          1. Premises first, followed by supported conclusions
          2. Intermediate conclusions as necessary for inferential clarity
       C. Document inferential pathways using the prescribed notation:
          `-- {{'from': ['x','y']}} --` indicating derivation from statements x and y
    
    III. Documentation Standards
       A. Begin with descriptive argument title
       B. Sequential labeling of all components
       C. Clear delineation of inferential steps
       D. Optional inclusion of argumentation schemes where applicable
    
    IV. Representational Constraints
       A. Single argument focus regardless of multiple threads in annotation
       B. Exclusion of:
          1. Dialectical relation indicators
          2. Propositional formalizations
          3. Extraneous metadata
    
    DOCUMENTATION FORMAT:
    Present the reconstruction in Argdown notation within a fenced code block (```argdown ... ```)
    
    This phase completes the analytical process by transforming the annotated textual data into a standardized argumentative structure for subsequent evaluation.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argument Reconstruction from Annotation
    
    ## Input
    Previously annotated argumentative text:
    ```
    {sources}
    ```
    
    ## Task Description
    Extract and reconstruct a single argument in standard form from the XML-annotated source.
    
    ## Input Format
    ```
    Type: XML with proposition annotations
    Key attributes: id, supports, attacks
    Relations: Support/attack relationships between elements
    ```
    
    ## Requirements
    
    ### Output Format
    ```
    Type: Argdown code block
    Delimiter: Triple backticks with language identifier
    ```
    
    ### Structural Requirements
    * Extract single primary argument from annotated text
    * Use proposition elements with "supports" attributes as primary source
    * Include:
      - Argument title/label
      - All necessary premises
      - Final conclusion 
      - Any intermediate conclusions
    
    ### Inference Documentation
    ```
    Format: -- {{'from': ['premise_id', 'premise_id']}} --
    Purpose: Document inferential paths between statements
    Placement: Immediately preceding each conclusion
    ```
    
    ### Constraints
    ```
    - Single argument reconstruction only
    - No dialectical relations in structure
    - No additional yaml data beyond 'from' attribute
    - No logical formalization
    ```
    
    ## Implementation Notes
    * Reference the proposition content from your annotation
    * Maintain logical flow from premises to conclusion
    * Ensure all referenced premises exist in the reconstruction
    * Use appropriate inference patterns based on annotation
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Creating an Argument Reconstruction from Your Annotation

    In this exercise, you'll transform your previous annotation into a structured argument. Let's break this down into simple steps:

    ## Step 1: Review Your Annotation
    First, carefully examine the annotation you've already created:

    ::: {{.source_text}}
    {sources}
    :::

    Pay special attention to:
    - The <proposition> elements you identified
    - Which propositions support others (check the "supports" attributes)
    - Which propositions attack others (check the "attacks" attributes)

    ## Step 2: Identify the Main Argument
    Look for the central claim being supported. This will typically be:
    - A proposition that is supported by several others
    - A proposition that represents the main point of the text
    - Not supporting many other propositions itself

    ## Step 3: Extract the Premises
    Find the propositions that directly or indirectly support your main conclusion:
    - Look at the "supports" attributes pointing to your conclusion
    - Consider if any premises support other premises (creating intermediate conclusions)

    ## Step 4: Organize into Standard Form
    Now, arrange these elements into a logical structure:
    1. Start with a title that captures the essence of the argument
    2. List all premises with clear numbering
    3. Show how premises connect to form conclusions

    ## Step 5: Document the Inference Steps
    For each conclusion, show which premises it comes from, e.g.:
    ```argdown
    // ...
    -- {{'from': ['2','3']}} --
    (4) Intermediate conclusion
    ```

    ## Step 6: Format Your Reconstruction
    Put your entire reconstruction in a code block, as shown in this minimal example:
    ```argdown
    <Title>: Main Argument Gist

    (1) First premise
    (2) Second premise
    -- {{'from': ['1','2']}} --
    (3) Conclusion
    ```

    ## Important Reminders:
    - Focus on just ONE main argument
    - Use your annotation as a guide, but express the premises clearly
    - Add implicit premises and conclusions if required
    - Only include the 'from' information in your YAML data
    - Don't add formal logic or symbolization

    Now, create your argument reconstruction based on the annotation!
    """).strip(),
    # Visualization-focused style
    dedent("""
    ARGUMENT EXTRACTION AND TRANSFORMATION REQUEST
    
    SOURCE MATERIAL (PREVIOUSLY ANNOTATED):
    ::: {{.source_text}}
    {sources}
    :::
    
    TASK: Transform the XML-annotated argumentative structure into a standardized premise-conclusion visualization using Argdown notation.
    
    VISUALIZATION PIPELINE:
    
    1. INPUT ANALYSIS
       • Source content: XML-annotated text with proposition elements
       • Structure indicators: support/attack relationships between elements
       • Primary focus: Elements with argumentative function (particularly support relations)
    
    2. EXTRACTION PARAMETERS
       • Target structure: Single coherent argumentative thread
       • Component identification: Premises and conclusions with inferential connections
       • Relation priority: Follow support relationships identified in annotation
       • Strictness: Deviate from annotation (omit / add / rephrase statements) to streamline argument flow

    3. TRANSFORMATION SPECIFICATIONS
       • Output format: Standard-form argument structure
       • Syntax: Argdown
       • Flow direction: Premises → intermediate steps → conclusion
       • Inference documentation: `-- {{'from': ['x','y']}} --` notation for all derived conclusions
       • Header: Descriptive title capturing argumentative essence
    
    4. VISUALIZATION FORMAT
       • Container: Fenced code block with language identifier (```argdown ... ```)
       • Component labeling: Sequential numeric identifiers
       • Structure: Linear progression from premises to conclusion
       • Metadata: Inference paths only, no additional YAML data
    
    5. STRUCTURAL CONSTRAINTS
       • Scope: Single argument extraction
       • Exclusions:
         - Dialectical relation indicators
         - Formal notation/symbolization
         - Multiple argument structures
    
    This visualization will transform the previously created annotation into a standardized argument structure for enhanced comprehension and analysis.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: From Annotation to Argument Structure
    
    In this tutorial, you'll learn how to transform your argumentative text annotation into a clear, structured argument in standard form.
    
    ## What You Have: An Annotated Text
    
    Let's look at the text you've already annotated with <proposition> tags:
    
    ::: {{.source_text}}
    {sources}
    :::
    
    ## What You'll Create: A Structured Argument
    
    You'll transform this annotation into a clear argument with premises and conclusions arranged to show how they support each other. You'll leave out non-essential branches, and uncover implicit premises.
    
    ## Why This Matters
    
    While annotation shows us where arguments exist in text, reconstructing them in standard form helps us:
    - See the logical structure more clearly
    - Evaluate whether conclusions actually follow from premises
    - Identify any missing steps or assumptions
    
    ## The Transformation Process
    
    ### Step 1: Study Your Annotation
    Look closely at:
    - Which propositions were you able to identify?
    - Which propositions support others? (check the "supports" attributes)
    - Which proposition seems to be the main claim? (often supported by many others)
    
    ### Step 2: Select the Main Argument
    Choose one central argument to reconstruct. Look for:
    - A significant claim with substantial support
    - A clear chain of reasoning
    - An important conclusion in the context of the text
    
    ### Step 3: Create Your Reconstruction
    Format your argument with:
    
    1. **A title** summarizing the main point
    2. **Numbered premises** (drawn from supporting propositions)
    3. **Inference lines** showing which statements lead to which conclusions:
       ```argdown
       // ...
       -- {{'from': ['1','2']}} --
       ```
    4. **Conclusions** (drawn from supported propositions)
    
    ### Example
    
    If your annotation showed proposition p1 supporting p3, and p2 supporting p3, your reconstruction might look like:
    
    ```argdown
    <Title>: Main Argument
    
    (1) /* Content from p1 */
    (2) /* Content from p2 */
    -- {{'from': ['1','2']}} --
    (3) /* Content from p3 */
    // ...
    ```
    
    ## Your Task
    
    Now, create your own argument reconstruction based on your annotation. Remember to:
    - Use the content from your annotated propositions
    - Include only one main argument
    - Format it properly in a code block with ```argdown at the start and ``` at the end
    - Show all inference relationships clearly
    
    Good luck!
    """).strip(),
]


class InfRecoFromArgAnnoProblem(InfRecoProblem):
    """
    Task: Reconstruct the main argument as a premise conclusion structure,
    no formalization, no dialectics.
    Input: Source text and argumentative annotation.
    """

    def __init__(self, annotation: str):
        if isinstance(annotation, list):
            annotation = "\n\n-----\n\n".join(annotation)
        # remove leading and trailing whitespace and newlines
        annotation = annotation.strip("\n ")
        self.annotation = annotation
        self.sources = annotation
        # randomly choose a prompt template
        self._prompt_template = random.choice(_INFRECO_FROM_ARGANNO_PROMPT_TEMPLATES)        


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(sources=self.annotation)
 
        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

        return prompt


class InfRecoFromArgAnnoProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arganno_pg = AnnotationProblemGenerator()
        self._arganno_sg = GenericSolutionGenerator(solution_class=Annotation, *args, **kwargs, n_solutions=10)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            arganno_problem = await self._arganno_pg.arun(inputs)
            arganno_solutions = await self._arganno_sg.arun(arganno_problem)
            arganno_evaluations =[
                AnnotationJudge()._evaluate_solution(s, arganno_problem)
                for s in arganno_solutions
            ]
            arganno_solution = next(
                (s for s, e in zip(arganno_solutions, arganno_evaluations) if e.is_valid),
                arganno_solutions[0]
            )
            return InfRecoFromArgAnnoProblem(str(arganno_solution))
        raise ValueError(
            "Inputs to an argument reconstruction problem must be a string or a list of strings"
        )


class AnnotationProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the source text's annotation."""

    hints = [
        "Make sure that your argument reconstruction stays faithful to and mimics closely "
        "the annotation of the source text. In particular, try to use supporting propositions from the annotation "
        "as premises / intermediate conclusions in your argument reconstruction!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, InfRecoFromArgAnnoProblem)
        assert isinstance(solution, InformalReco)
        assert isinstance(solution.argdown_snippet, str)

        soup, _ = AnnotationJudge().parse_xml_snippet(problem.annotation)
        anno_props = soup.find_all("proposition")
        supporting_props = [
            ap
            for ap in anno_props
            if ap.get("supports")  # type: ignore
        ]
        list_anno_props = "\n".join([ap.text for ap in supporting_props])

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                list_anno_props, solution.argdown_snippet
            ),
            1,
        )
