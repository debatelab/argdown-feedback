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
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    InfRecoCompositeHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.content_check_handler import HasArgdownHandler
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)


_INFRECO_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
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
    - Do NOT add any formalization of the argument's propositions (premises or conclusions) in your Argdown code.
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be argument builders! 🏗️

    I need your help to reconstruct the hidden argument in this text and put it together like a puzzle!

    Here's the text to look at:

    ::: {{.source_text}}
    {sources}
    :::

    Your mission is to find the MAIN argument and show how it fits together with premises (reasons) and a conclusion (what the author wants you to believe).

    Here's how to complete your mission:

    1. First, find all the important reasons (premises) in the text
    2. Then find what the author is trying to convince us of (conclusion)
    3. Show how the reasons connect to make the conclusion (using intermediate steps if needed)

    When you write down the argument:
    - Put everything in a special code box that starts with ```argdown and ends with ```
    - Give your argument a cool title at the top
    - Number each reason (premise)
    - For each conclusion, show which reasons it comes from using this special code: `-- {{'from': ['1','3']}} --` (this means the conclusion comes from reasons 1 and 3)
    - Only show ONE main argument (not lots of different arguments)

    Important things NOT to do:
    - Don't include any extra arguments or maps
    - Don't add any special symbols like + or -> between the premises
    - Don't add any extra code besides the 'from' information
    - Don't try to make the argument super fancy with symbols or equations

    I know you can do this! Good luck, builder!
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! Could you help me break down the main argument in this text?

    Check this out:

    ::: {{.source_text}}
    {sources}
    :::

    I need to reconstruct the central argument in standard form using Argdown syntax. Nothing too complex - just looking for the core reasoning.

    If you could:
    - Put your reconstruction in a code block (start with '```argdown' ... and end with '```')
    - Focus on just ONE main argument (which may include some intermediate steps)
    - For each conclusion, note which premises it builds from using `-- {{'from': ['1','4']}} --` format (meaning it's derived from premises 1 and 4)
    - Add a clear title for the argument at the beginning
    
    A few things to avoid:
    - No need for multiple arguments or argument maps
    - Skip any inline dialectical relations (like support/attack indicators)
    - Keep the yaml data simple - just stick to the 'from' information
    - No formal logic symbols needed
    
    Basically looking for a clean, informal breakdown of the main argument flow. Thanks so much!
    """).strip(),
    # Academic style
    dedent("""
    ASSIGNMENT: Argument Reconstruction and Analysis (Standard Form)
    
    OBJECTIVE: Informally reconstruct the principal argument presented in the source text utilizing standard form and Argdown notation.
    
    SOURCE MATERIAL:
    ::: {{.source_text}}
    {sources}
    :::
    
    REQUIREMENTS:
    
    1. STRUCTURE AND FORMAT
       • Present your reconstruction as a premise-conclusion structure
       • Enclose the entirety of your reconstruction in a properly demarcated code block (```argdown ... ```)
       • Designate a concise, descriptive title for the argument at the outset of your reconstruction
       
    2. INFERENCE DOCUMENTATION
       • For each conclusion (intermediary or final), explicitly document its inferential basis
       • Utilize the following YAML inline data notation: `-- {{'from': ['3','4']}} --`
       • Ensure label references correspond accurately to previously established premises or intermediate conclusions
       
    3. OPTIONAL ELEMENTS
       • You may, at your discretion, include notation regarding inference rules or argumentation schemes
       
    4. CONSTRAINTS
       • Limit your reconstruction to a single, coherent argument
       • Omit dialectical relations within the premise-conclusion structure
       • Refrain from including extraneous YAML data beyond required inference documentation
       • Exclude any formalized representation of propositions
    
    EVALUATION CRITERIA:
    Your submission will be assessed based on accuracy of reconstruction, adherence to format specifications, and appropriate identification of inferential relationships.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Protocol: Argument Reconstruction in Standard Form
    
    OBJECTIVE:
    To extract and reconstruct the principal argumentation structure from the provided source text utilizing standard form argument notation with Argdown syntax.
    
    SOURCE TEXT FOR ANALYSIS:
    ::: {{.source_text}}
    {sources}
    :::
    
    METHODOLOGICAL REQUIREMENTS:
    
    I. Reconstruction Parameters
       A. Isolate a singular, coherent argumentative thread from the source text
       B. Render the argument in standard premise-conclusion format using Argdown syntax
       C. Include all relevant premises and sub-conclusions leading to the final conclusion
    
    II. Documentation Standards
       A. For each inferential step, document the precise premises from which it is derived
       B. Implement the following notation for inference tracking:
          `-- {{'from': ['2','3']}} --` (indicating derivation from premises 2 and 3)
       C. Taxonomically identify argumentation schemes if discernible (optional)
    
    III. Presentation Format
       A. Preface the reconstruction with a concise, descriptive argument title
       B. Enclose the entire reconstruction within a fenced code block (```argdown ... ```)
       C. Maintain clear premise and conclusion labeling throughout
    
    METHODOLOGICAL CONSTRAINTS:
    - Exclude extraneous argumentative analyses beyond the central reconstruction
    - Omit dialectical relation indicators within the premise-conclusion structure
    - Restrict YAML metadata exclusively to inferential source documentation
    - Avoid logical notation or formalization of propositional content
    
    This protocol serves to ensure consistent extraction of argumentative structures for subsequent analytical evaluation.
    """).strip(),
    # Developer-focused style
    dedent("""
    // Task: Reconstruct a source text's main argument in standard form using Argdown syntax
    // Objective: Extract the primary argument and represent it as a premise-conclusion structure
    
    /**
    * @input - Raw text document containing argumentative content (SOURCE TEXT)
    * @output - Argdown code block representing the main argument in standard form
    * @format - Fenced code block with language identifier 'argdown'
    * @structure - Single argument with premises, conclusions, and inference paths
    * @constraints - No dialectical relations, no formal logic notation, no extraneous YAML data
    * @optional - Inference rules or argumentation schemes may be included
    */
    
    SOURCE TEXT:
    
    :::
    {sources}
    :::
    
    Implementation Notes:
    * Extract single primary argument only
    * Include:
      - Argument title/label
      - All premises (properly labeled)
      - Final conclusion 
      - Any intermediate conclusions
    * Document inference paths using YAML inline data:
      ```
      -- {{'from': ['<premise-id>','<premise-id>']}} --
      ```
    * Fenced code block format:
      ```argdown
      <title>
           
      <premise-conclusion-structure>
      ```           
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Informal Argument Reconstruction Guide: Standard Form

    Let's break down the main argument in this text step by step:

    ::: {{.source_text}}
    {sources}
    :::

    ## Step 1: Identify the Components
    First, read the text carefully and identify:
    - The main conclusion (what the author is trying to prove)
    - The premises (reasons given to support the conclusion)
    - Any intermediate conclusions (points that are both supported by some premises and support the main conclusion)

    ## Step 2: Create the Structure
    Now, let's organize these into a standard argument form:

    1. Start with a title that captures the essence of the argument
    2. List all premises with clear numbering
    3. Show how premises connect to form intermediate conclusions
    4. Show how everything leads to the final conclusion

    ## Step 3: Document the Inference Steps
    For each conclusion (intermediate or final), show which premises it comes from using this format:
    `-- {{'from': ['3','5']}} --` (meaning the conclusion below this inference line follows from premises 3 and 5)

    ## Step 4: Format Your Reconstruction
    Put your entire reconstruction in a code block:
    ```argdown
    // Your reconstruction goes here
    ```

    ## Important Reminders:
    - Focus on just ONE main argument
    - Don't add support/attack relations between premises
    - Only include the 'from' information in your YAML data
    - Keep everything in plain language (no formal logic symbols)
    - Adhere to Argdown syntax conventions

    I'm looking forward to seeing your clear, well-structured argument reconstruction!
    """).strip(),
    # Visualization-focused style
    dedent("""
    ARGUMENT EXTRACTION REQUEST
    
    SOURCE CONTENT FOR ANALYSIS:
    ::: {{.source_text}}
    {sources}
    :::
    
    TASK: Extract the central argumentative structure from the provided text and render it in standard premise-conclusion format using Argdown notation.
    
    EXTRACTION SPECIFICATIONS:
    
    1. STRUCTURAL ELEMENTS
       • Core argument identification: Extract the single most significant argument
       • Component delineation: Clearly distinguish premises and conclusions
       • Inferential pathways: Document precise derivation paths between components
    
    2. VISUALIZATION PARAMETERS
       • Format: Premise-conclusion standard form
       • Notation: Argdown syntax
       • Metadata: For each conclusion, specify source premises using `-- {{'from': ['x','y']}} --` format
       • Header: Include descriptive argument title and summary
    
    3. TECHNICAL REQUIREMENTS
       • Container: Fenced code block with language identifier (```argdown ... ```)
       • Component labeling: Sequential numeric identifiers
       • Information flow: Clearly traceable from premises to final conclusion
    
    4. CONSTRAINTS
       • Scope: Single argument extraction only
       • Exclusions:
         - Dialectical relation indicators
         - Extraneous YAML metadata
         - Formal notation/symbolization
         - Multiple argument structures

    This reconstruction will serve as input to an argument analysis system for subsequent evaluation and processing.
    """).strip(),
    # Tutorial style
    dedent("""
    # Exercise: Reconstructing an Argument in Standard Form

    In this exercise, I ask you to reconstruct an argument from natural text into standard form using Argdown notation.

    ## The Source Text

    First, let's examine our text:

    ::: {{.source_text}}
    {sources}
    :::

    ## What is Standard Form?

    Standard form presents an argument as a series of premises leading to a conclusion, showing the logical structure clearly.

    ## Your Task

    You'll reconstruct the **main argument** from this text following these guidelines:

    ### Required Elements:
    1. **Title** - A brief descriptor of the argument
    2. **Premises** - The supporting reasons
    3. **Conclusion** - The main point being argued for
    4. **Inference Structure** - How premises connect to conclusions

    ### The Process:

    1. **Identify the conclusion** first (what is the author trying to prove?)
    2. **Work backwards** to find the premises that support it
    3. **Arrange** premises and conclusions in logical order
    4. **Document** which premises support each conclusion using:
       `-- {{'from': ['4','5']}} --` (meaning "from premises 4 and 5")

    ### Generic example illustrating the format:

    ```argdown
    <Title>

    (1) First premise
    (2) Second premise
    -- {{'from': ['1','2']}} --
    (3) Optional intermediate conclusion
    (4) Additional premise
    -- {{'from': ['3','4']}} --
    (5) Final conclusion
    ```
           
    Of course, number of premises and conclusions may vary!

    ### Common Mistakes to Avoid Here:
           
    - Including multiple arguments (focus on just one)
    - Adding dialectical relations in the structure
    - Including extra YAML data
    - Using formal logic symbols

    Now it's your turn! Informally reconstruct the argument in standard form.
    """).strip(),
]


class InfRecoProblem(Problem):
    """Task: Reconstruct the main argument as a premise conclusion structure, no formalization, no dialectics."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace and newlines
        sources = sources.strip("\n ")
        self.sources = sources
        # randomly choose a prompt template
        self._prompt_template = random.choice(_INFRECO_PROMPT_TEMPLATES)

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
class InformalReco(Solution):
    """Solution to the argument analysis problem: an argdown snippet."""

    argdown_snippet: str
    _raw_answer: str

    def __str__(self):
        return self.argdown_snippet

    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else self.argdown_snippet

    @classmethod
    def from_raw_answer(cls, raw_answer) -> "InformalReco":
        """extract the argdown snippet from a raw answer"""
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)
        code_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )
        code_snippet = code_snippet if code_snippet is not None else raw_answer
        return cls(argdown_snippet=code_snippet, _raw_answer=raw_answer)


class InfRecoProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return InfRecoProblem(inputs)
        raise ValueError(
            "Inputs to an argument recinstruction problem must be a string or a list of strings"
        )


class InfRecoJudge(MPJudge):
    """Judge for the informal argument reconstruction task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(original_solution, InformalReco) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(isinstance(solution, InformalReco) for solution in solutions), (
            "All solutions must be InformalReco objects"
        )

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"

        infreco_handler = InfRecoCompositeHandler()
        # remove UsesAllPropsHandler
        infreco_handler.handlers = [
            h
            for h in infreco_handler.handlers
            if not isinstance(h, UsesAllPropsHandler)
        ]
        handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                ArgdownParser(name="ArgdownParser"),
                HasArgdownHandler(),
                infreco_handler,
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


class InfRecoFeedbackGenerator(FeedbackGenerator):
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
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
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


class NoUnusedPropsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument reconstruction task, prefering valid recos
    with fewer unused premises or conclusions."""

    hints = [
        "In your argument reconstruction, make sure that every premise and every intermediate conclusion is "
        "(explicitly) used in a subsequent inference. (Every unused premise or conclusion counts as a mistake.)"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        argument = argdown.arguments[0]
        used_labels = set()
        for c in argument.pcs:
            if isinstance(c, Conclusion):
                used_labels.update(c.inference_data.get("from", []))
        number_unused_props = sum(
            1 for p in argument.pcs[:-1] if p.label not in used_labels
        )

        return (number_unused_props + 1) ** -1


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
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
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
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
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
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
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
        assert isinstance(problem, InfRecoProblem), "Problem must be an InfRecoProblem"
        assert isinstance(solution, InformalReco), "Solution must be an InformalReco"
        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, solution.argdown_snippet
            ),
            1,
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
