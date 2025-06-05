import random
from textwrap import dedent
from pyargdown import Argdown
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
    InfRecoProblemGenerator,
    InfRecoJudge,
    InformalReco,
)
from argdown_feedback.tasks.core.logreco import (
    LogicalReco,
    LogRecoProblem,
)


_LOGRECO_FROM_INFRECO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
    Assignment: Logically analyse an informally reconstructed argument.
                        
    Your task is to logically analyze and, as necessary, revise, the following informally reconstructed argument using Argdown.
    Formalize all the premises and conclusions in your revision. Make sure the revised argument is deductively valid and all 
    premises are relevant.

    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    Note in particular:

    - Enclose your Argdown argument reconstruction in a fenced codeblock, starting with '```argdown' and
      ending with '```'. Just include a single Argdown codeblock in your answer.

    - In your Argdown snippet, only include *a single argument* which re-analyzes the informal reconstruction (adding/removing or revising 
      premises, final conclusion, and possible intermediate conclusions as required).

    - For each proposition in your reconstruction (premises and conclusions), provide an adequate propositional logic / FOL formalization in NLTK
      syntax. Use yaml inline data with keys 'formalization' and 'declarations' to record your logical analyses. Minimal example:
      `(1) Socrates is mortal. {{formalization: 'F(a)', declarations: {{'a': 'Socrates', 'F': 'being mortal'}} }}`.
      Only declare variables that are used in the corresponding formalization and that have not been declared before.
      Ensure that your formalizations are consistent with each other.

    - For each inference step in the argument, provide information about which previously introduced premises or 
      conclusions it uses. Indicate this via yaml inline data with key 'from' in the inference line, e.g. `-- {{'from': ['2','3']}} --`,
      where the list items refer to the respective premise or conclusion labels.
    
    - You may, but are in no way required to add additional information about which inference rules or argumentation
      schemes are applied in each sub-argument.

    - In addition, at the beginning of your Argdown code block, provide a succinct label (title) for the argument and 
      summarize its gist in line with Argdown syntax conventions. 
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be logic detectives! 🕵️‍♀️🔍

    I have an argument that someone already wrote down, but I need YOUR help to make it super clear using logic!

    Here's the argument we need to work with:

    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    Your mission is to:

    1. Look at the argument and understand what it's trying to say
    2. Add special logic formulas to each part of the argument
    3. Make sure the whole argument is valid (that means the conclusion really follows from the premises!)
    4. Fix anything that needs fixing to make the argument work properly

    Here's how to complete your mission:

    - Put your new version of the argument in a special code box that starts with ```argdown and ends with ```
    - For EACH sentence in the argument, add a special formula using logic symbols
    - Use this format for your logic formulas: {{formalization: 'YOUR_FORMULA', declarations: {{'p': 'what p stands for', 'q': 'what q stands for'}} }}
    - For any conclusion, show which previous statements it comes from using: `-- {{'from': ['1','2']}} --` (this means the conclusion comes from statements 1 and 2)
    - Give your argument a cool title at the top

    Remember to use NLTK syntax for your logic formulas, like we learned before!

    I know you'll do an amazing job making this argument clear and logical!
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! I've got this informal argument reconstruction that needs to be formalized with proper logical notation.

    Take a look at this argument:

    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    What I need you to do is add logical formalization to this argument and make sure it's actually valid. You know how sometimes informal arguments have gaps or don't quite connect logically? Your job is to fix that and turn this into a proper deductive argument.

    Here's how to approach it:

    - Take the basic structure from the informal reconstruction but feel free to revise as needed
    - For each statement (premises and conclusions), add a formal logic representation using propositional logic / FOL in NLTK syntax
    - Format it like this: {{formalization: 'p -> q', declarations: {{'p': 'proposition represented by p', 'q': 'proposition represented by q'}} }}
    - Show which premises lead to each conclusion with `-- {{'from': ['1','3']}} --` notation
    - Make sure the argument is actually valid when formalized (conclusion must necessarily follow from premises)
    - Only include premises that are actually needed

    Put your complete formalized argument in a code block starting with ```argdown and ending with ```. And don't forget to give it a good title at the beginning!

    Thanks for your help with this - looking forward to seeing your logical analysis!
    """).strip(),
    # Academic style
    dedent("""
    ASSIGNMENT: Logical Analysis and Formalization of Informal Argument

    OBJECTIVE: Provide a rigorous logical formalization and analysis of the provided informal argument reconstruction, ensuring deductive validity and premise relevance.

    SOURCE ARGUMENT:
    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    REQUIRED TASKS:

    1. LOGICAL ASSESSMENT
       • Evaluate the informal argument for structural soundness
       • Identify any inferential gaps, ambiguities, or logical fallacies
       • Determine necessary revisions to ensure deductive validity
       
    2. FORMALIZATION PROTOCOL
       • Render each proposition (premises and conclusions) in Propositional / First-Order Logic
       • Employ NLTK syntax for all logical expressions
       • Provide explicit declarations for all non-logical symbols
       • Example format: {{formalization: '(x).(P(x) -> Q(x))', declarations: {{'P': 'being P', 'Q': 'being Q'}} }}
       
    3. STRUCTURAL REQUIREMENTS
       • Maintain or revise the basic argumentative structure as necessary
       • Add implicit premises or intermediate conclusions where required
       • Document inferential pathways using standardized notation: `-- {{'from': ['premise_numbers']}} --`
       • Ensure all premises contribute substantively to the conclusion (principle of relevance)
       • Adhere to Argdown syntax conventions
       
    4. PRESENTATIONAL FORMAT
       • Begin with a concise descriptive title
       • Enclose the formalized reconstruction within a delimited code block (```argdown ... ```)
       • Maintain consistent numbering throughout
       • Optional: Include references to applicable logical rules or argumentation schemes

    EVALUATION CRITERIA:
    Your formalization will be assessed on logical validity, formal accuracy, and Argdown syntax adherence.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Sequential Logical Analysis - Phase II (Formalization of Informal Reconstruction)
    
    PROCEDURAL CONTEXT:
    This protocol outlines the second phase of a sequential argument analysis procedure, in which an informal argument reconstruction is subjected to logical formalization and deductive validation.
    
    SOURCE MATERIAL (INFORMAL RECONSTRUCTION):
    {argdown_snippet}
    
    ANALYTICAL OBJECTIVES:
    To derive a formally valid logical structure from the provided informal reconstruction while maintaining maximum fidelity to the original argumentative intent, while handling any flaws in the informal reconstruction gracefully.
    
    METHODOLOGICAL REQUIREMENTS:
    
    I. Assessment Parameters
       A. Evaluate inferential adequacy of the informal reconstruction
       B. Identify logical gaps or inconsistencies
       C. Determine necessary structural modifications
    
    II. Formalization Specifications
       A. Propositional Analysis
          1. Formalize each statement in propositional / first-order logic (NLTK syntax)
          2. Provide comprehensive declaration of all non-logical symbols
          3. Example format: {{formalization: 'p v -q', declarations: {{'p': 'proposition represented by p', 'q': 'proposition represented by q'}} }}

       B. Inferential Documentation
          1. Specify precise derivational pathways for each conclusion
          2. Implement standardized notation: `-- {{'from': ['x','y']}} --`
          3. Optional: Document applicable inference rules
    
    III. Structural Requirements
       A. Maintain core argumentative thread
       B. Implement necessary modifications to ensure:
          1. Deductive validity
          2. Premise relevance
          3. Logical consistency
       C. Add implicit premises or intermediate conclusions as required
    
    IV. Documentation Standards
       A. Begin with descriptive argument title
       B. Maintain consistent propositional numbering
       C. Ensure consistent variable usage across formalizations
       D. Preserve inferential transparency
    
    OUTPUT FORMAT:
    Present the formalized reconstruction in Argdown notation within a delimited code block (```argdown ... ```)
    
    This protocol ensures systematic transformation of informal argumentative structures into formally validated logical representations suitable for rigorous evaluation.
           
    Submit your formalized argument reconstruction in the specified format below, adhering to all outlined requirements.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Logical Formalization Task: InfReco to LogReco

    ## Input
    Informal argument reconstruction:
    ```
    {argdown_snippet}
    ```

    ## Task Description
    Transform the informal argument reconstruction into a logically formalized version with propositional logic / FOL notation.

    ## Input Format
    ```
    Type: Argdown informal reconstruction
    Structure: Premise-conclusion format with inference paths
    ```

    ## Output Requirements

    ### Structural Requirements
    ```
    - Maintain or revise basic argument structure
    - Ensure deductive validity
    - Include only relevant premises
    - Add implicit premises if necessary
    ```

    ### Formalization Requirements
    ```
    // For each proposition:
    (n) Natural language statement {{
      formalization: 'well_formed_formula',
      declarations: {{
        'symbol': 'meaning',
        'symbol': 'meaning',
      }}
    }}
    ```

    ### Technical Specifications
    * Use NLTK syntax for Propositional / First Order Logic expressions
    * Document inference paths with `-- {{'from': ['premise_id', 'premise_id']}} --`
    * Include argument title at beginning
    * Maintain logical consistency across all formalizations
    * Ensure symbol declarations are complete and non-redundant

    ### Output Format
    ```argdown
    // Complete formalized argument
    ```

    ## Validation Criteria
    * All statement formalized in NLTK syntax 
    * All symbols properly declared
    * All inference paths documented
    * Deductive validity according to formalizations and inference tree
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Formalizing an Informal Argument: Step-by-Step Guide

    In this exercise, you'll transform an informal argument reconstruction into a formalized logical version. Let's break this down into manageable steps:

    ## Step 1: Understand the Original Argument
    First, carefully review the informal argument:

    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    Pay special attention to:
    - The overall structure
    - The main conclusion
    - The supporting premises
    - The inference relationships

    ## Step 2: Plan Your Logical Analysis
    Consider:
    - Is the argument deductively valid as presented?
    - Are there implicit premises needed?
    - Are all the premises relevant?
    - What logical form would best represent each statement?

    ## Step 3: Formalize Each Statement
    For each premise and conclusion:
    1. Express it in natural language (you can use or revise the original wording)
    2. Create a formal logical representation using Propositional / First Order Logic with NLTK syntax
    3. Add declarations explaining what each symbol means

    For example:
    ```argdown 
    // ...
    (1) All humans are mortal. {{formalization: '(x).(Human(x) -> Mortal(x))', declarations: {{'Human': 'being human', 'Mortal': 'being mortal'}} }}
    ```

    ## Step 4: Document the Inference Structure
    For each conclusion, specify which premises it follows from:
    ```argdown
    // ...
    -- {{'from': ['1','2']}} --
    (3) Socrates is mortal. // ...
    ```

    ## Step 5: Review for Validity
    Check that:
    - When formalized, the conclusion logically follows from the premises
    - Your formalizations are consistent with each other
    - All variables and predicates are properly declared

    ## Step 6: Format Your Answer
    Put your complete formalized argument in a code block, as in this minimal example:
    ```argdown
    <Title>: A Clear Title for Your Argument

    (1) First premise... {{formalization: '...', declarations: {{...}} }}
    (2) Second premise... {{formalization: '...', declarations: {{...}} }}
    -- {{'from': ['1','2']}} --
    (3) Conclusion... {{formalization: '...', declarations: {{...}} }}
    ```

    Remember: Focus on making the argument logically valid while staying as faithful as possible to the original informal reconstruction.
    """).strip(),
    # Visualization-focused style
    dedent("""
    LOGICAL STRUCTURE FORMALIZATION REQUEST

    SOURCE MATERIAL:
    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    OBJECTIVE: Transform the provided informal argument reconstruction into a formalized logical structure with explicit symbolic representation.

    FORMALIZATION PARAMETERS:

    1. LOGICAL ARCHITECTURE
       • Base structure: Derived from source informal reconstruction
       • Validity requirement: Ensure deductive validity of all subinferences
       • Component modification: Add/revise premises or conclusions as necessary
       • Symbolic representation: Propositional /first-order logic with NLTK syntax

    2. SYMBOLIC NOTATION SYSTEM
       • Statement formalization: Each natural language proposition translated to well-formed logical form
       • Symbol declaration: Every non-logical symbol explicitly defined
       • Format specification:
         ```
         (n) Natural language statement {{formalization: 'logical_formula', 
            declarations: {{'symbol': 'meaning'}} }}
         ```
       • Cross-statement consistency: Maintain uniform symbol usage across formulas

    3. INFERENTIAL PATHWAYS
       • Documentation format: `-- {{'from': ['premise_numbers']}} --`
       • Placement: Immediately preceding each derived conclusion
       • Validation: Ensure each step represents a valid deductive inference
       • Optional enhancement: Add logical rules applied

    4. STRUCTURAL PRESENTATION
       • Heading: Descriptive title capturing argumentative essence
       • Sequence: Logical progression from premises through inference steps to conclusion
       • Encapsulation: Complete structure within fenced code block (```argdown ... ```)
       • Visual clarity: Consistent formatting throughout

    COMPLETION CRITERIA:
    The formalized structure must represent a deductively valid argument with consistent logical notation, complete symbol declarations, and explicit inference pathways.
           
    Subnmit your formalized argument reconstruction in the specified format below, adhering to all outlined requirements.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Adding Logical Formalization to an Argument

    In this tutorial, you'll learn how to transform an informal argument into a formally valid logical argument using symbolic notation.

    ## What You're Starting With

    Here's an informal argument reconstruction:

    ::: {{.informal_argument}}              
    {argdown_snippet}
    :::

    ## What You'll Create

    You'll transform this into a logically formalized argument where:
    - Each statement has a formal logical representation
    - The logical structure is explicitly shown
    - The argument and all its subarguemnts are deductively valid

    ## Why This Matters

    Formalizing arguments helps us:
    - Identify hidden assumptions and gaps in reasoning
    - Test if conclusions really follow from premises
    - Express complex ideas with precision and clarity

    ## The Transformation Process

    ### 1. Analyze the Informal Argument
    
    First, understand what the argument is claiming and how it's structured. Note:
    - What's the main conclusion?
    - What premises support it?
    - Are there any gaps in the reasoning?

    ### 2. Plan Your Formalization
    
    For each statement, decide:
    - What logical form best represents it? (universal statement, conditional, etc.)
    - What predicates and variables will you need?
    - Are additional premises needed for validity?

    ### 3. Formalize Each Statement
    
    For each statement:
    - Keep or streamline the natural language expression
    - Add a formalization using First Order Logic in NLTK syntax
    - Add declarations explaining what each symbol means

    Example:
    ```argdown 
    // ...
    (2) If Joe quits, the company goes bankrupt. {{formalization: 'p -> q', declarations: {{'p': 'Joe quits', 'q': 'the company goes bankrupt'}} }}
    ```

    ### 4. Show Inference Relationships
    
    For each conclusion, show which premises it follows from:
    ```argdown
    // ...
    -- {{'from': ['1','3']}} --
    (4) Therefore, robins have feathers.
    ```

    ## Minimal Example

    Here's a simple example of how an informal argument becomes formalized:

    Informal:
    ```argdown
    (1) All mammals are warm-blooded
    -- {{'from': ['1']}} --
    (2) Whales are warm-blooded
    ```

    Formalized:
    ```argdown
    (1) All mammals are warm-blooded. {{formalization: '(x). (M(x) -> W(x))', declarations: {{'M': 'being a mammal', 'W': 'being warm-blooded'}} }}
    (2) Whales are mammals. {{formalization: '(x). (Wh(x) -> M(x))', declarations: {{'Wh': 'being a whale'}} }}
    -- {{'from': ['1','2']}} --
    (3) Whales are warm-blooded. {{formalization: '(x). (Wh(x) -> W(x))', declarations: {{}} }}
    ```

    ## Your Task

    Now, create a formalized version of the informal argument we've been starting with above. Remember to:
    - Give it a clear title
    - Formalize each statement
    - Declare all symbols
    - Show inference relationships
    - Ensure the argument is valid when formalized
    - Put everything in a code block with ```argdown at the start and ``` at the end

    Good luck with your formalization!
    """).strip(),
]


class LogrecoFromInfrecoProblem(LogRecoProblem):
    """
    Task: Logically analyse, revise and formalize the informally reconstructed argument.
    Input: infreco.
    """

    def __init__(
        self,
        argdown_snippet: str,
        argdown_infreco: Argdown | None = None,
        infreco_evaluation: Evaluation | None = None,
    ):
        argdown_snippet = argdown_snippet.strip()
        self.argdown_snippet = argdown_snippet
        self.argdown_infreco = argdown_infreco
        self.infreco_evaluation = infreco_evaluation
        self.sources = argdown_snippet
        # randomly choose a prompt template
        self._prompt_template = random.choice(_LOGRECO_FROM_INFRECO_PROMPT_TEMPLATES)

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(argdown_snippet=self.argdown_snippet)

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)

        return prompt


class LogrecoFromInfrecoProblemGenerator(ProblemGeneratorLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._infreco_pg = InfRecoProblemGenerator()        
        self._infreco_sg = GenericSolutionGenerator(solution_class=InformalReco, *args, **kwargs, n_solutions=1)

    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            infreco_problem = await self._infreco_pg.arun(inputs)
            infreco_solution = await self._infreco_sg.arun(infreco_problem)            
            infreco_evaluation = InfRecoJudge()._evaluate_solution(infreco_solution[0], infreco_problem)
            argdown_infreco = infreco_evaluation.artifacts.get("argdown_reco")
            return LogrecoFromInfrecoProblem(
                argdown_snippet=str(infreco_solution[0]),
                argdown_infreco=argdown_infreco,
                infreco_evaluation=infreco_evaluation,
            )
        raise ValueError(
            "Inputs to an LogrecoFromInfrecoProblem must be a string or a list of strings"
        )


class InfrecoProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the original informal reconstruction."""

    hints = [
        "Make sure that your logical analysis stays faithful to and mimics closely "
        "the original informal reconstruction. In particular, try to re-use premises and conclusions "
        "where possible, or apply minimal changes to the informal reconstruction.",
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        reco = solution
        assert isinstance(problem, LogrecoFromInfrecoProblem)
        assert isinstance(reco, LogicalReco)

        argdown_infreco = problem.argdown_snippet
        argdown_logreco = reco.argdown_snippet

        if not argdown_infreco or not argdown_logreco:
            return 0.0

        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                argdown_infreco, argdown_logreco
            ),
            1,
        )
