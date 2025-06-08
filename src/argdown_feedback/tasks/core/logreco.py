import random
from typing import Sequence

import dataclasses
from textwrap import dedent
import textdistance

from pyargdown import (
    ArgdownMultiDiGraph,
    Conclusion,
    Proposition,
)

from argdown_feedback.tasks.base import (
    MPJudge,
    Problem,
    ScoringVirtuePreferencePairGenerator,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    FeedbackGenerator,
)

from argdown_feedback.logic.logic import get_propositional_variables
from argdown_feedback.logic.fol_to_nl import FOL2NLTranslator
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import InfRecoCompositeHandler, NoPropInlineDataHandler
from argdown_feedback.verifiers.core.logreco_handler import LogRecoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import HasArgdownHandler
from argdown_feedback.verifiers.processing_handler import ArgdownParser, DefaultProcessingHandler, FencedCodeBlockExtractor
from argdown_feedback.verifiers.verification_request import VerificationDType, VerificationRequest


_LOGRECO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
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

    - Do NOT include any other analyses (maps or arguments) in your Argdown snippet besides the reconstruction of the main argument.
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there, young logician! Today we're going to be logic detectives! 🔍🧩

    I need your help to find the hidden argument in this text and make it super clear using logic symbols!
           
    Remember Argdown? It's the special way we write arguments so we can easily check them.

    Here's the text to look at:

    ::: {{.source_text}}
    {sources}
    :::

    Your mission is to:
    1. Find the MAIN argument in the story
    2. Write it down as premises and a conclusion
    3. Turn each sentence into a special logic formula (this is the cool part!)

    Here's how to complete your mission:

    1. First, find all the important reasons (premises) in the text
    2. Then find what the author is trying to convince us of (conclusion)
    3. Show how the reasons connect to make the conclusion, adding intermediate conclusions if needed
    4. For EACH sentence, create a logic formula!

    When you write down the argument:
    - Put everything in a special code box that starts with ```argdown and ends with ```
    - Give your argument a cool title at the top
    - Number each reason (premise)
    - After each sentence, add a special "formalization" like this: 
      {{formalization: 'P(x)', declarations: {{'x': 'thing being described', 'P': 'what we're saying about it'}} }}
    - Use the NLTK notation we've practiced last week for logic formulas
    - For each conclusion, show which reasons it comes from using this special code: `-- {{'from': ['1','3']}} --` (this means the conclusion comes from reasons 1 and 3)

    Remember: This is like turning the argument into a math problem that we can solve!

    I know you can do this! Good luck, logic detective!
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! I could use your help turning this text into a formal logical argument.

    Check out this passage:

    ::: {{.source_text}}
    {sources}
    :::

    I need to reconstruct the main argument in a way that's logically valid, with everything formalized properly. Here's what I'm looking for:

    - Find the main argument and put it in standard form (premises leading to conclusion, possibly via intermediate steps)
    - For each statement, add a formal logic representation using propositional logic / FOL in NLTK syntax
    - Make sure the whole thing is deductively valid (conclusion necessarily follows)
    - Only include premises that are actually needed

    If you could format it like this:
    - Put everything in an argdown code block (```argdown ... ```)
    - Start with a title that captures the essence of the argument
    - Number each premise clearly
    - After each statement, include {{formalization: 'logical formula', declarations: {{variables explained}} }}
    - Show which premises lead to each conclusion with `-- {{'from': ['1','3']}} --`

    Just focus on one main argument - make sure it's valid and the formalizations are consistent. Thanks!
    """).strip(),
    # Academic style
    dedent("""
    FORMAL RECONSTRUCTION ASSIGNMENT: Logical Formalization Exercise

    OBJECTIVE: Provide a logically valid reconstruction of the central argument presented in the source text, complete with formal logical notation.

    SOURCE TEXT:
    ::: {{.source_text}}
    {sources}
    :::

    REQUIREMENTS:

    1. STRUCTURAL COMPONENTS
       • Identify and isolate a single, central argument from the source text
       • Present the argument in standard logical form (premises → intermediate conclusions → final conclusion)
       • Ensure deductive validity (conclusion must necessarily follow from premises)
       • Include only necessary premises (principle of relevance)

    2. FORMALIZATION PROTOCOL
       • For each proposition, provide:
         a) Natural language articulation
         b) First-order logic formalization in NLTK syntax
         c) Variable/predicate/constant declarations with clear natural language referents
       • Example: (1) All mammals are vertebrates. {{formalization: '(x).(M(x) -> V(x))', declarations: {{'M': 'being a mammal', 'V': 'being a vertebrate'}} }}

    3. INFERENTIAL DOCUMENTATION
       • Document each inference step with explicit notation of premise dependencies
       • Utilize YAML inline data format: `-- {{'from': ['premise-numbers']}} --`
       • Example: `-- {{'from': ['1','2']}} --`

    4. PRESENTATIONAL FORMAT
       • Enclose the entire reconstruction within a fenced code block (```argdown ... ```)
       • Begin with a descriptive title summarizing the argument's central claim
       • Maintain sequential labeling of all premises and conclusions
       • Ensure typographical consistency throughout the reconstruction

    EVALUATION CRITERIA:
    Your submission will be assessed on logical validity, formalization accuracy, inferential clarity, and adherence to specified formatting conventions.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Logical Reconstruction and Formalization (LRF-9)
                        
    OBJECTIVE:
    To extract, reconstruct, and formalize the principal argumentative structure from the provided source text, ensuring deductive validity and logical soundness.

    SOURCE MATERIAL:
    ::: {{.source_text}}
    {sources}
    :::

    METHODOLOGICAL FRAMEWORK:

    I. Extraction Phase
       A. Identify the central argumentative thread within the source material
       B. Isolate all explicit and implicit premises essential to the argument's structure
       C. Determine the main conclusion toward which the reasoning progresses

    II. Reconstruction Protocol
       A. Arrange identified propositions in standard form using Argdown syntax
       B. Include only relevant premises (eliminate extraneous content)
       C. Add necessary intermediate conclusions to ensure transparent inferential pathways
       D. Ensure deductive validity of the overall argument structure and each inferential step

    III. Formalization Requirements
       A. For each proposition:
          1. Express in natural language 
          2. Provide first-order logic formalization using NLTK syntax
          3. Include comprehensive declarations of placeholders
          4. Example format: {{formalization: 'P(a) -> Q(a))', declarations: {{'P': 'property P', 'Q': 'property Q', 'a': 'object a'}} }}
       
       B. For each inference:
          1. Document precisely which premises/conclusions support each inferential step
          2. Use standardized notation: `-- {{'from': ['1','3']}} --`
          3. Optional: Add inference rule identification

    IV. Documentation Guidelines
       A. Begin with concise argument title and overview
       B. Enclose entire formalization in specified code block format (```argdown ... ```)
       C. Ensure consistent numerical labeling throughout
       D. Maintain internal consistency among all logical formalizations

    This protocol document serves as the operational standard for logical argument extraction and formalization procedures.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Logical Argument Reconstruction Task

    ## Input Specification
    Source text containing argumentative content:
    ```
    {sources}
    ```

    ## Task Description
    Extract and reconstruct the main argument as a deductively valid logical structure with formal notation.

    ## Output Requirements

    ### Format Requirements
    ```
    Content-Type: text/argdown
    Enclosure: Triple backticks with language identifier
    ```

    ### Structural Requirements
    * Extract single primary argument only
    * Ensure deductive validity
    * Include only relevant premises
    * Structure components:
      - Argument title/label
      - All required premises
      - Final conclusion
      - Any necessary intermediate conclusions

    ### Formalization Requirements
    * For each proposition:
      ```
      (n) Natural language statement {{
        formalization: 'FOL_expression', 
        declarations: {{
          'placeholder symbol': 'substitution',
        }}
      }}
      ```
    * Use NLTK syntax for propositional logic / FOL expressions
    * Maintain consistent variable/predicate usage across formalizations

    ### Inference Documentation
    * Format: `-- {{'from': ['premise_id', 'premise_id']}} --`
    * All conclusions must have explicit inference paths
    * Optional: Include inference rules or argumentation schemes

    ## Constraints
    ```
    - Single argument reconstruction only
    - No dialectical relations within structure
    - No extraneous analyses or commentary
    - Full formalization required for all propositions
    ```

    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Logical Argument Reconstruction Guide

    Let's transform an ordinary text into a formal logical argument step by step:

    First, read this text carefully:

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Find the Main Argument
    Look for the central claim being defended and the reasons given to support it.

    ## Step 2: Extract the Premises and Conclusion
    Identify all the important statements that form the argument.

    ## Step 3: Arrange in Standard Form
    Put the premises first, followed by the conclusion they support.

    ## Step 4: Add Logical Formalization
    For each statement in your reconstruction:
    1. Write the statement in natural language
    2. Add a formalization using propositional / first-order logic
    3. Define what each symbol means

    Example:
    ```
    (1) All birds have feathers. {{formalization: 'AllX(B(x) -> F(x))', declarations: {{'B': 'being a bird', 'F': 'having feathers'}} }}
    ```

    ## Step 5: Connect the Inferences
    Show which premises lead to each conclusion using:
    ```
    -- {{'from': ['1','2']}} --
    ```

    ## Step 6: Check Your Work
    Make sure:
    - The argument is deductively valid (conclusion necessarily follows)
    - All premises are actually needed
    - Formalizations are consistent across the argument
    - All variables and predicates are properly declared

    ## Step 7: Format Your Final Answer
    Put everything in a code block:
    ```argdown
    Title: [Give your argument a clear title]

    (1) First premise... {{formalization: '...', declarations: {{...}} }}
    (2) Second premise... {{formalization: '...', declarations: {{...}} }}
    -- {{'from': ['1','2']}} --
    (3) Conclusion... {{formalization: '...', declarations: {{...}} }}
    ```

    Remember: Focus on making your argument logically valid with accurate formalizations!
    """).strip(),
    # Visualization-focused style
    dedent("""
    FORMAL ARGUMENT STRUCTURE VISUALIZATION REQUEST

    SOURCE CONTENT:
    ::: {{.source_text}}
    {sources}
    :::

    TASK SPECIFICATION: Extract the central argumentative structure from the provided text and render it as a logically formalized argument in standard form.

    FORMALIZATION PARAMETERS:

    1. STRUCTURAL ELEMENTS
       • Argument identification: Extract single most significant argument
       • Propositional layout: Standard premise-conclusion format
       • Logical validity: Ensure deductive validity across all inferences
       • Premise selection: Include only relevant premises

    2. LOGICAL NOTATION SCHEMA
       • Formalism: First-order predicate logic with NLTK syntax
       • Transparency: Include natural language interpretation of all logical symbols
       • Component format:
         ```
         (n) Natural language statement {{formalization: 'logical_formula', 
            declarations: {{'symbol': 'meaning'}} }}
         ```

    3. INFERENCE ARCHITECTURE
       • Document inference pathways explicitly:
         `-- {{'from': ['premise_numbers']}} --`
       • Ensure logical soundness at each inferential step
       • Optional: Include logical rules/schemes used

    4. VISUAL REPRESENTATION PARAMETERS
       • Format: Argdown code block (```argdown ... ```)
       • Hierarchy: Sequential arrangement from premises to conclusion
       • Labeling: Clear numerical identifiers for each component
       • Title: Descriptive header encapsulating central claim

    OUTPUT REQUIREMENTS:
    The formalized argument must adhere precisely to the specifications above, with consistent logical notation throughout and explicit mapping between natural language components and their logical counterparts.

    This formalization will serve as input for advanced logical analysis and argument evaluation systems.
    """).strip(),
    # Tutorial style
    dedent("""
    # Tutorial: Reconstructing an Argument with Logical Formalization

    In this exercise, you'll learn how to extract an argument from natural text and represent it formally with logical notation.

    ## The Source Text

    First, let's examine our text:

    ::: {{.source_text}}
    {sources}
    :::

    ## What is Logical Reconstruction?

    Logical reconstruction means:
    1. Identifying the core argument in a text
    2. Arranging it in standard form (premises → conclusion)
    3. Formalizing each statement using logical notation
    4. Ensuring the argument is deductively valid

    ## Your Task

    You'll reconstruct the **main argument** from this text and add formal logical notation:

    ### Step 1: Identify Components
    Find:
    - The main conclusion (what the author ultimately wants to prove)
    - The premises (reasons given to support the conclusion)
    - Any implicit premises needed for validity

    ### Step 2: Create the Structure
    1. Start with a title that captures the essence of the argument
    2. List all premises with clear numbering
    3. Show how premises connect to form conclusions

    ### Step 3: Add Formal Notation
    For each statement, add:
    - A FOL formalization using NLTK syntax
    - Declarations explaining what each symbol means

    Example:
    ```
    (1) All humans are mortal. {{formalization: 'AllX(H(x) -> M(x))', declarations: {{'H': 'being human', 'M': 'being mortal'}} }}
    ```

    ### Step 4: Document Inference Steps
    For each conclusion, show which premises it comes from:
    ```
    -- {{'from': ['1','2']}} --
    (3) Intermediate conclusion
    ```

    ## Format Your Answer
    Put your complete reconstruction in an Argdown code block:
    ```argdown
    Title: The Main Argument

    (1) First premise {{formalization: '...', declarations: {{...}} }}
    (2) Second premise {{formalization: '...', declarations: {{...}} }}
    -- {{'from': ['1','2']}} --
    (3) Final conclusion {{formalization: '...', declarations: {{...}} }}
    ```

    Remember: Your reconstruction should be deductively valid, meaning the conclusion necessarily follows from the premises when formalized properly.

    Now it's your turn to create a logical reconstruction!
    """).strip(),
]
    


class LogRecoProblem(Problem):
    """Task: Reconstruct the main argument as deductively valid using premise conclusion structure and including formalization."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace and newlines
        sources = sources.strip("\n ")
        self.sources = sources
        # randomly choose a prompt template
        self._prompt_template = random.choice(_LOGRECO_PROMPT_TEMPLATES)


    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(sources=self.sources)

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_prompt(prompt, evaluation)


        return prompt

    def revise_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = "Revise your previously submitted argument reconstruction given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class LogicalReco(Solution):
    """Solution to the argument analysis problem: an argdown snippet."""

    argdown_snippet: str
    _raw_answer: str

    def __str__(self):
        return self.argdown_snippet
    
    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else self.argdown_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer) -> "LogicalReco":
        """Extract a LogicalReco from a raw answer string."""
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)
        code_snippet = next(
            (
                vr.code_snippet for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )
        code_snippet = code_snippet if code_snippet is not None else raw_answer
        return cls(argdown_snippet=code_snippet, _raw_answer=raw_answer)
    

class LogRecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return LogRecoProblem(inputs)
        raise ValueError(
            "Inputs to an argument reconstruction problem must be a string or a list of strings"
        )


class LogRecoJudge(MPJudge):
    """Judge for the informal argument reconstruction task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(original_solution, LogicalReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        for solution in solutions:
            assert isinstance(solution, LogicalReco), (
                "All solutions must be LogicalReco objects"
            )


    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:

        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(solution, LogicalReco), "Solution must be an LogicalReco"

        infreco_handler = InfRecoCompositeHandler()
        infreco_handler.handlers = [
            h for h in infreco_handler.handlers if not isinstance(h, NoPropInlineDataHandler)
        ]

        handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                ArgdownParser(name="ArgdownParser"),
                HasArgdownHandler(),
                infreco_handler,
                LogRecoCompositeHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=solution.argdown_snippet, source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_reco") is None:
            evaluation.artifacts["argdown_reco"] = evaluation.artifacts.get("argdown")
        return evaluation


class LogRecoFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_feedbacks", 5)
        self.gen_kwargs = kwargs.get("gen_kwargs", {"max_tokens": 1024})

    async def arun(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> list[Feedback]:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(solution, LogicalReco), "Solution must be an LogicalReco"
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given argument reconstruction.

            You will be shown an argument analysis problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

                                                
            ## Problem Statement
            {problem}

            
            ## Student's Solution
            {solution}

            
            ## Evaluation
            The student's solution is NOT valid.
            Particular issues:
            {evaluation_issues}

            
            Given this information, provide feedback to the student and clear instructions for how to improve the solution.
        """).format(
            problem=problem.instruct_prompt(),
            solution=str(solution),
            evaluation_issues=evaluation_issues,
        )

        answers = await self._generate(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            n=self.n_feedbacks,
            **self.gen_kwargs,
        )
        # remove empty and duplicate answers
        answers = [a for a in answers if a]
        answers = list(set(answers))

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class ManyIntermediateConclusionsPreferencePairGenerator(
    ScoringVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with more intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to include as many sub-arguments as possible. "
        "I.e., reconstruct the argument with many intermediate steps. That is what counts here."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return number_intermediate_conclusions


class FewIntermediateConclusionsPreferencePairGenerator(
    ScoringVirtuePreferencePairGenerator
):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with fewer intermediate conclusions."""

    hints = [
        "In your argument reconstruction, try to minimize the number of intermediate conclusions. "
        "I.e., reconstruct the argument with as few sub-arguments as possible. That is what counts here."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        number_intermediate_conclusions = sum(
            1 for p in argument.pcs[:-1] if isinstance(p, Conclusion)
        )

        return (number_intermediate_conclusions + 1) ** -1


class IndependentWordingPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with independent wording of arguments and claims."""

    hints = [
        "Make sure that you render the argument's premises and conclusion(s) *in your own words*, "
        "and independently from the formulations in the source text. This is crucial at this step."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        dlds: list[float] = []
        for p in propositions:
            for t in p.texts:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_distance(
                        problem.sources, t
                    )
                )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0


class SourceTextProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco task, prefering valid argument recos
    that stick closely to the source text."""

    hints = [
        "Make sure that your argument reconstruction stays maximally faithful to and mimics closely the original source text!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, LogRecoProblem), "Problem must be an LogRecoProblem"
        assert isinstance(solution, LogicalReco), "Solution must be an LogicalReco"
        return round(
                textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, solution.argdown_snippet
            ),
            1
        )


class SimplicityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with succinct and simple propositions."""

    hints = [
        "Make sure that you keep each of the argument's premises and conclusion(s) simple and succinct. "
        "Short sentences are crucial at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) ** -1 if lengths else 0
    

class VerbosityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with elaborate and verbose propositions."""

    hints = [
        "Render the argument's premises and conclusion(s) in an elaborate and verbose way. "
        "Long sentences are strongly preferred at this step. (Number of premises and conclusions is not important.)"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        propositions: list[Proposition] = argdown.propositions

        lengths: list[float] = []
        for p in propositions:
            for t in p.texts:
                lengths.append(len(t))

        return round(sum(lengths) / len(lengths), -1) if lengths else 0



class FormalizationsFaithfulnessPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with formalizations that are similiar to the sentences being formalized."""

    hints = [
        "Reconstruct the argument in such a way that your logico-semantic analysis (formalizations and declarations) "
        "coheres with the actual wording of the premises and conclusion(s). In particular, formalize your argument's "
        "premises and conclusion(s) faithfully!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        all_expressions = evaluation.artifacts["all_expressions"]
        all_declarations = evaluation.artifacts["all_declarations"]

        dlds: list[float] = []
        for pr in argument.pcs:
            expression = all_expressions.get(pr.proposition_label)
            proposition = argdown.get_proposition(pr.proposition_label)

            if expression is None or proposition is None:
                continue 

            text_1 = FOL2NLTranslator.translate_to_nl_sentence(
                expression, all_declarations
            )

            for text_2 in proposition.texts:
                #print(f"Comparing '{text_1}' and '{text_2}'")
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_similarity(
                        text_1, text_2
                    )
                )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0


class PredicateLogicPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reco, prefering valid reconstructions
    with formalizations that use but predicate logic."""

    hints = [
        "Formalize the premises and conclusions in your argument reconstruction "
        "using predicate logic. Avoid using propositional logic! No propositional variables!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        all_expressions = evaluation.artifacts["all_expressions"]
        if not all_expressions:
            return 0
        n_has_prop_vars = sum(bool(get_propositional_variables(expr)) for expr in all_expressions.values())
        return 1 - (n_has_prop_vars / len(all_expressions))



