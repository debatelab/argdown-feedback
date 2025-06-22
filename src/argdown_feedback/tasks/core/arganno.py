import dataclasses
import random
from textwrap import dedent

from bs4 import BeautifulSoup

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

from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler, SourceTextIntegrityHandler
from argdown_feedback.verifiers.core.content_check_handler import HasAnnotationsHandler
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)
from argdown_feedback.verifiers.processing_handler import (
    FencedCodeBlockExtractor,
    XMLParser,
)
from argdown_feedback.verifiers.base import CompositeHandler

ANNOTATION_SCHEME = dedent("""
    <!ELEMENT proposition   (#PC-DATA)                          -- single element marking a (sub-)sentence involved in the argumentation -->
    <!ATTLIST proposition   id              ID      #REQUIRED   -- unique id of element -->
    <!ATTLIST proposition   supports        IDREFS  #IMPLIED    -- other (sub-)sentences supported or confirmed by this element (empty space separated) -->
    <!ATTLIST proposition   attacks         IDREFS  #IMPLIED    -- other (sub-)sentences attacked or disconfirmed by this element (empty space separated) -->
    <!ATTLIST proposition   argument_label  CDATA   #IMPLIED    -- unique label of argument or thesis in external argdown document -->
    <!ATTLIST proposition   ref_reco_label  CDATA   #IMPLIED    -- unique item label of premise or conclusion in external argdown argument -->
""")

_ANNOTATION_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
        Assignment: Apply a given annotation scheme to a source text.
                    
        Annotate the following **source text** in order to identify the argumentative function of different parts in the text.

        ::: {{.source_text words={word_count}}}
        {sources}
        :::

        Annotate the source text above according to the following schema:

        {annotation_scheme}

        Just add tags and attributes to the source text to mark the argumentative function of each part. Don't modify the text in any other way (exception: non-annotated segments of long texts with more than {word_count_threshold} words may be shortened).

        Enclose the annotated text in a single fenced codeblock, starting with '```xml' and ending with '```'.
        """).strip(),
    # elementary school style
    dedent("""
        Hello there, my smart AI helper! Today we're going to play a fun detective game with some text!

        Your assignment: Put on your detective hat and look for clues in a piece of text!
                    
        Please read this **story** below very carefully:

        ::: {{.source_text words={word_count}}}
        {sources}
        :::

        Now, I want you to be a text detective! Using the special tags below, mark the parts of the story that make arguments:

        {annotation_scheme}

        Remember, dear AI: Just add the tags around the words - don't change any of the words in the story! (If the story is super long and has more than {word_count_threshold} words, you can shorten parts that don't have arguments.)

        When you're done, put your marked-up text between these special markers:
        ```xml at the beginning
        ``` at the end

        You're such a clever helper! I know you'll do a wonderful job with this! 🌟
        """).strip(),
    dedent("""
        Hey there, could you help me out with something when you get a minute? I've got this annotation task that needs doing.
                    
        So basically, I need you to go through this text document and mark up the argumentative parts:

        ::: {{.source_text words={word_count}}}
        {sources}
        :::

        The boss wants us to use this specific annotation scheme - I know it looks complicated but you're good at this stuff:

        {annotation_scheme}

        Just add those tags where needed but don't change any of the actual wording. If there are long sections with no arguments and the word count exceeds {word_count_threshold}, you can abbreviate those parts.

        When you're done, could you put the whole thing in one of those code blocks? Start with ```xml and end with ```. Makes it easier to process later. -- Don't forget the final ``` at the end! 

        Thanks so much for this - I owe you a coffee!
        """).strip(),
    dedent("""
        Task: Argumentative Structure Annotation (25 points)
                        
        Analyze the following passage by identifying its argumentative structure. Apply the appropriate XML annotation tags to demarcate propositions and their logical relationships.

        ::: {{.source_text words={word_count}}}
        {sources}
        :::

        Annotation Requirements:
        • Use the formal annotation scheme as specified in class:
        
        {annotation_scheme}
        
        • Add XML tags to identify propositions and their argumentative relationships
        • Clearly indicate support and attack relationships between propositions
        • Maintain the integrity of the original text (though you may abbreviate non-argumentative passages in source texts with more than {word_count_threshold} words)
        • Ensure each proposition has a unique identifier
        
        Submission Format:
        Your annotated text must be enclosed within a code block demarcated by ```xml at the beginning and ``` at the end, i.e.,

        ```xml
        (your annotated text here)
        ```
        
        Note: Careful attention to the logical relationships between claims will be essential for full credit.
        """).strip(),
    dedent("""
        Good day, AI assistant. I require your analytical capabilities for an annotation task related to my current research.
        
        I'm working with a text that needs precise argumentative annotation for an upcoming publication. Could you assist me in applying an annotation scheme to identify argumentative structures?

        Here's the passage I'm analyzing:

        ::: {{.source_text words={word_count}}}
        {sources}
        :::

        Please apply the following XML-based annotation scheme that my colleagues and I have developed for our research:
        
        {annotation_scheme}
        
        For methodological rigor, please adhere to the following protocols:
        1. Mark all argumentative propositions with appropriate tags
        2. Preserve the exact wording of the original text
        3. Identify all support and attack relationships between propositions
        4. Assign unique identifiers to each proposition
        5. You may abbreviate non-argumentative passages in very long texts (> {word_count_threshold} words) if necessary
        
        Present your analysis in a machine-readable format by enclosing the annotated text in a code block with ```xml at the beginning and ``` at the end.

        Thank you for your assistance with this scholarly endeavor. This will significantly advance my research timeline.
        """).strip(),
    dedent("""
        Hey AI, I need to process this text for an argument analysis tool I'm building.
        
        // Task: Parse text and identify argument structures using the XML schema below
        
        /**
        * @input - Raw text document containing argumentative content (SOURCE TEXT)
        * @output - XML-annotated version with proposition tags and relationships
        */
        
        Implementation Notes:
        - Mark all propositions with appropriate tags
        - Each proposition needs a unique ID attribute
        - Link related propositions using supports/attacks attributes
        - Maintain the original text content (do not modify wording)
        - Non-argumentative sections can be abbreviated with [...] notation if word count exceeds {word_count_threshold}
        
        SOURCE TEXT ({word_count} words):
        ::: {{.source_text}}
        {sources}
        :::
        
        /* Schema definition - implement this exactly */
        {annotation_scheme}
        
        // Return annotated text as valid XML within code block
        // Format: ```xml at start, ``` at end
        
        This is for a critical module in my project - thanks for the help!
        """).strip(),        
    dedent("""
        Please help me in understanding the following TEXT:

        TEXT ({word_count} words):
        :::
        {sources}
        :::

        I fail to see how the argumentation is structured in the TEXT:
           
        - Which propositions have an argumentative function?
        - How are they related to each other?

        I came across the following scheme (XML annotation scheme):

        {annotation_scheme}

        ... and these implementation notes:
        - Mark all propositions with appropriate tags
        - Each proposition needs a unique ID attribute
        - Link related propositions using supports/attacks attributes
        - Maintain the original text content (do not modify wording)
        - Non-argumentative sections in texts with more than {word_count_threshold} words can be abbreviated with [...] notation
        - Embed the annotated text within a code block (```xml at start, ``` at end)

        Please provide the annotated text in the specified format.
        """).strip(),
    dedent("""
        CODEBOOK: ARGUMENTATIVE STRUCTURE ANNOTATION PROTOCOL
        Version 1.2
        
        SECTION 1: OVERVIEW AND PURPOSE
        This protocol guides the systematic identification and annotation of argumentative structures within texts. The annotation process involves marking propositions and their logical relationships using a standardized XML tagging scheme.
        
        SECTION 2: SOURCE MATERIAL
        Please annotate the following text according to the protocol described below:
        
        ::: {{.source_text words={word_count}}}
        {sources}
        :::
        
        SECTION 3: ANNOTATION SCHEMA
        Apply the following formal annotation scheme to identify argumentative elements:
        
        {annotation_scheme}
        
        SECTION 4: CODING PROCEDURES
        4.1 Identification Phase
        - Identify all propositions that serve an argumentative function
        - Assign unique identifier attributes to each proposition
        - Preserve exact wording of the original text
        
        4.2 Relationship Coding
        - For each proposition, determine if it supports or attacks other propositions
        - Record support relationships using the "supports" attribute (space-separated IDs)
        - Record attack relationships using the "attacks" attribute (space-separated IDs)
        
        4.3 Technical Guidelines
        - Non-argumentative segments in texts with more than {word_count_threshold} words may be abbreviated with [...] notation
        - Do not modify any text within identified propositions
        - Ensure all XML tags are properly nested and closed
        
        SECTION 5: SUBMISSION FORMAT
        Submit the completed annotation as a single XML document enclosed in a code block:
        - Begin with ```xml
        - End with ```
        
        NOTE: Intercoder reliability assessment will be conducted using automated verification tools.
        """).strip(),
]


class AnnotationProblem(Problem):
    """Task: Apply the argumentative annotation scheme to a text."""

    def __init__(self, sources: str | list[str], strip_html: bool = True):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # strip html tags
        if strip_html:
            sources = BeautifulSoup(sources, "html.parser").get_text()
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources
        # randomly choose a prompt template
        self._prompt_template = random.choice(_ANNOTATION_PROMPT_TEMPLATES)

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = self._prompt_template.format(
            sources=self.sources,
            annotation_scheme=ANNOTATION_SCHEME,
            word_count=len(self.sources.split()),
            word_count_threshold=SourceTextIntegrityHandler._ALLOW_SOURCE_TEXT_SHORTENING_WC_THRESHOLD,
        )

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
        prompt = (
            "Revise your previous annotation given the above evaluation and feedback."
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class Annotation(Solution):
    """Solution to the annotation problem: just an annotated text."""

    annotated_source_text: str | None = None
    _raw_answer: str | None = None

    def __str__(self):
        if self.annotated_source_text is not None:
            return self.annotated_source_text
        return self._raw_answer if self._raw_answer is not None else "None"

    def raw_answer(self) -> str:
        """Returns the full and raw answer as a string, including any reasoning traces"""
        return self._raw_answer if self._raw_answer else str(self)

    @classmethod
    def from_raw_answer(cls, raw_answer) -> "Annotation":
        """Extract the annotated source text from the answer."""
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=raw_answer)
        result = handler.process(request)
        code_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.xml and vr.code_snippet
            ),
            None,
        )
        return cls(annotated_source_text=code_snippet, _raw_answer=raw_answer)


class AnnotationProblemGenerator(ProblemGenerator):
    # TODO: Vary and configure the annotation problems generated
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return AnnotationProblem(inputs)
        raise ValueError(
            "Inputs to an annotation problem must be a string or a list of strings"
        )


class AnnotationJudge(MPJudge):
    """Judge for the annotation task."""

    def parse_xml_snippet(
        self, annotated_source_text: str
    ) -> tuple[BeautifulSoup, str | None]:
        error_msg: str | None = None
        ast = annotated_source_text.strip("\n ")
        if ast.startswith("```xml") and ast.endswith("```") and len(ast.splitlines()) > 1:
            ast = "\n".join(ast.splitlines()[1:-1])
        else:  # no fenced code block
            error_msg = "Failed to extract single fenced annotation block:"
            if ast.count("```xml") == 0:
                error_msg += " No fenced code block starting with '```xml'."
            if ast.count("```xml") > 1:
                error_msg += " More than one fenced code block starting with '```xml'."
            if not ast.endswith("```"):
                error_msg += " No closing '```'."

        multi_valued_attributes = {"*": {"supports", "attacks"}}
        soup = BeautifulSoup(
            ast,
            "html.parser",
            multi_valued_attributes=multi_valued_attributes,
        )
        return soup, error_msg

    def _check_inputs(self, problem, solutions, original_solution = None, feedback = None):
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(original_solution, Annotation) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        for solution in solutions:
            assert isinstance(solution, Annotation), "All solutions must be Annotations"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(solution, Annotation), "Solution must be an Annotation"

        handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                XMLParser(name="XMLAnnotationParser"),
                HasAnnotationsHandler(),
                ArgannoCompositeHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=str(solution), source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        return evaluation
    


class AnnotationFeedbackGenerator(FeedbackGenerator):
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
        assert isinstance(problem, AnnotationProblem), (
            "Problem must be an AnnotationProblem"
        )
        assert isinstance(solution, Annotation), "Solution must be an Annotation"
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given annotation.

            You will be shown an argumentative annotation problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

                                                
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


class AnnotationScopePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of annotated proposition elements."""

    hints = ["Try to identify as many proposition elements as possible"]

    def _score(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> float:
        return len(evaluation.artifacts["soup"].find_all("proposition"))


class AnnotationSupportsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of support relations between propositions."""

    hints = ["Try to identify as many support relations as possible"]

    def _score(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        supports = sum(
            len(proposition.get("supports", [])) for proposition in propositions
        )
        return supports


class AnnotationAttacksPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger number of attack relations between propositions."""

    hints = ["Try to identify as many attack / disconfirmation relations as possible"]

    def _score(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", [])) for proposition in propositions
        )
        return attacks


class AnnotationNoAttacksPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with smallest number of attack relations between propositions."""

    hints = ["Avoid using attack / disconfirmation relations"]

    def _score(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        attacks = sum(
            len(proposition.get("attacks", [])) for proposition in propositions
        )
        return 1 / (1 + attacks)


class AnnotationCoveragePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid annotations
    with larger coverage of source text."""

    hints = ["Try to cover as much of the source text as possible"]

    def _score(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> float:
        propositions = evaluation.artifacts["soup"].find_all("proposition")
        coverage = sum(len(proposition.get_text()) for proposition in propositions)
        return coverage
