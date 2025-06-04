from typing import Sequence

import dataclasses
from textwrap import dedent
import textdistance

import networkx as nx  # type: ignore
from pyargdown import (
    ArgdownEdge,
    ArgdownMultiDiGraph,
    Argument,
    Proposition,
    Valence,
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
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import HasArgdownHandler
from argdown_feedback.verifiers.processing_handler import (
    ArgdownParser,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)



_ARGMAP_PROMPT_TEMPLATES = [
    # Default template
    dedent("""
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

    Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.                                                
    """).strip(),
    # Elementary school style
    dedent("""
    Hello there! Today we're going to be argument detectives! 🕵️‍♀️🕵️‍♂️
                
    I want you to read this story carefully and find all the arguments hidden inside:

    ::: {{.source_text}}
    {sources}
    :::

    Now, let's make an argument map using Argdown! Here's how:

    1. Find all the main ideas (we'll call them "claims")
    2. Find all the reasons people give (we'll call them "arguments")
    3. Draw lines to show which arguments support or attack which claims

    Remember these special rules:
    - Give every idea a clear label
    - Put square brackets [ ] around claim labels
    - Put angled brackets < > around argument labels
    - Use arrows (+ and -) to show which ideas support or attack other ideas
    
    Please don't write out all the details of each argument - just the key points and main connections!

    When you're finished, put your argument map inside a special box:
    ```argdown
    (your map goes here)
    ```

    I can't wait to see what you discover! 🌟
    """).strip(),
    # Casual/friendly style
    dedent("""
    Hey there! Mind helping me map out the arguments in this text?
                
    I'm trying to understand the reasoning structure in this passage:

    ::: {{.source_text}}
    {sources}
    :::

    Could you create an Argdown argument map that shows how everything connects? Nothing too fancy - just:

    - Label each argument and claim clearly
    - Use square brackets [like this] for claims and angled brackets <like this> for arguments
    - Show which arguments support or attack which claims using Argdown's syntax
    
    Just focus on the big picture connections between arguments - no need to break down each argument into premises and conclusions.

    When you're done, just drop your map in a code block starting with ```argdown and ending with ```.

    Thanks a ton for your help with this!
    """).strip(),
    # Academic style
    dedent("""
    Argumentative Structure Analysis Assignment
                
    INSTRUCTIONS: Conduct a thorough analysis of the argumentative structure present in the following text by constructing an Argdown argument map that accurately represents the dialectical relationships between claims and arguments.

    SOURCE TEXT:
    ::: {{.source_text}}
    {sources}
    :::

    REQUIREMENTS:
    1. Identify and explicitly label all argumentative nodes (both claims and arguments)
    2. Employ proper syntactic conventions:
       a. Utilize square brackets for claim labels
       b. Utilize angled brackets for argument labels
    3. Accurately represent the dialectical relations (support/attack) between nodes
    4. Adhere to Argdown syntactic conventions for representing these relations
    
    CONSTRAINTS:
    Do not include detailed argument reconstructions as premise-conclusion structures.

    SUBMISSION FORMAT:
    Present your argument map within a fenced code block demarcated by triple backticks and the argdown language identifier (```argdown) at the beginning and triple backticks (```) at the conclusion.

    NOTE: Your grade will be determined by the accuracy and comprehensiveness of your argumentative reconstruction.
    """).strip(),
    # Research-oriented style
    dedent("""
    Research Task: Argument Mapping for Dialectical Analysis
                
    OBJECTIVE: To produce a formal representation of the argumentative structure embedded in the provided source text using Argdown notation.

    SOURCE MATERIAL:
    ::: {{.source_text}}
    {sources}
    :::

    METHODOLOGY:
    1. Conduct a thorough argumentation analysis of the provided text
    2. Identify all relevant claims and arguments constituting the dialectical structure
    3. Represent this structure as an Argdown argument map

    TECHNICAL SPECIFICATIONS:
    - All nodes must be explicitly labeled for clarity and reference
    - Distinguish between claims and arguments using syntactic conventions:
      * Claims: [square bracket notation]
      * Arguments: <angled bracket notation>
    - Document all inferential and dialectical relationships between nodes using standard Argdown conventions
    - Focus on macro-level argumentation relations rather than micro-level argument structure
    - Avoid detailed premise-conclusion breakdowns within arguments
    
    OUTPUT FORMAT:
    The resulting argument map must be enclosed within a fenced code block using appropriate markup (```argdown to begin, ``` to end).

    NOTE: This analysis will inform subsequent computational processing for automated reasoning assessment.
    """).strip(),
    # Developer-focused style
    dedent("""
    # Argdown Argument Map Generation
    
    ## Input
    Process the following text for argumentative content:
    
    ```
    {sources}
    ```
    
    ## Requirements
    
    Generate an argument map in Argdown syntax that meets the following specifications:
    
    * Extract all claims and arguments from source
    * Label format:
      - Claims: [square_brackets]
      - Arguments: <angled_brackets>
    * Document all support/attack relationships
    * Focus on graph structure, not internal argument composition
    
    ## Expected Output
    
    Return a valid Argdown map enclosed in a code block:
    
    ```argdown
    // Your argument map here
    ```
    
    ## Notes
    
    * Do not include premise-conclusion structures within arguments
    * Ensure consistent labeling across all nodes
    * Follow standard Argdown syntax conventions for directional relationships
    """).strip(),
    # Step-by-step guidance style
    dedent("""
    # Argument Mapping Exercise
    
    Let's create an Argdown argument map from this text:
    
    ::: {{.source_text}}
    {sources}
    :::
    
    ## Step 1: Identify Key Claims
    First, identify all key claims in the text. These are the main points being argued for or against.
    
    ## Step 2: Identify Arguments
    Next, identify the arguments that are being made to support or attack these claims.
    
    ## Step 3: Create Labels
    For each claim and argument, create a descriptive label:
    - Use [square brackets] for claim labels
    - Use <angled brackets> for argument labels
    
    ## Step 4: Map Relationships
    Show how arguments and claims relate to each other:
    - Use `+>`/`<+` to show support relationships
    - Use `->`/`<-` to show attack relationships

    ## Step 5: Format Your Map
    Put your completed map in a code block:
    ```argdown
    // Your argument map goes here
    ```
    
    Remember: Focus on the relationships between arguments and claims - don't break down any argument into premises and conclusions.
    """).strip(),
    # Visualization-focused style
    dedent("""
    VISUALIZATION REQUEST: Argument Network Structure
                
    Please generate a textual representation of the argument network contained within the following source material:

    ::: {{.source_text}}
    {sources}
    :::

    REPRESENTATION FORMAT: Argdown Notation

    The visualization should fulfill these criteria:

    1. NODE IDENTIFICATION
       • Extract all argumentative nodes (claims and arguments)
       • Apply descriptive labels to each node
       • Format: [square_brackets] for claims, <angled_brackets> for arguments

    2. EDGE REPRESENTATION
       • Map support relationships between nodes
       • Map attack relationships between nodes
       • Adhere to Argdown directional syntax conventions

    3. STRUCTURAL FOCUS
       • Prioritize inter-argument relationships
       • Omit internal argument structures (premise-conclusion details)

    4. OUTPUT FORMAT
       • Enclose in fenced code block (```argdown ... ```)
       • Ensure valid Argdown syntax for processing

    This representation will serve as input to a visual argument mapping tool.
    """).strip(),
]


class ArgMapProblem(Problem):
    """Task: Map the arguments in a text as an informal argdown argument map."""

    def __init__(self, sources: str | list[str]):
        if isinstance(sources, list):
            sources = "\n\n-----\n\n".join(sources)
        # remove leading and trailing whitespace
        sources = sources.strip()
        self.sources = sources

    def instruct_prompt(
        self,
        ask_for_invalid=False,
        hints: list[str] | None = None,
        evaluation: Evaluation | None = None,
    ) -> str:
        prompt = (
            dedent("""
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

            Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.                                                
            """)
            .strip()
            .format(sources=self.sources)
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
        prompt = "Revise your previously submitted argument map given the above evaluation and feedback."

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt = self.ask_for_invalid_revise_prompt(prompt, evaluation)

        return prompt


@dataclasses.dataclass
class ArgumentMap(Solution):
    """Solution to the argument mapping problem: an argdown snippet."""

    argdown_snippet: str

    def __str__(self):
        return self.argdown_snippet

    @classmethod
    def from_raw_answer(cls, answer) -> "ArgumentMap":
        # extract fenced code block
        handler = FencedCodeBlockExtractor()
        request = VerificationRequest(inputs=answer)
        result = handler.process(request)
        code_snippet = next(
            (
                vr.code_snippet
                for vr in reversed(result.verification_data)
                if vr.dtype == VerificationDType.argdown and vr.code_snippet
            ),
            None,
        )
        code_snippet = code_snippet if code_snippet is not None else answer
        return cls(argdown_snippet=code_snippet)


class ArgMapProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgMapProblem(inputs)
        raise ValueError(
            "Inputs to an argument mapping problem must be a string or a list of strings"
        )


class ArgMapJudge(MPJudge):
    """Judge for the argument mapping task."""

    def _check_inputs(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> None:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(original_solution, ArgumentMap) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )
        assert all(
            isinstance(solution, ArgumentMap) for solution in solutions
        ), "All solutions must be ArgumentMaps"

    @staticmethod
    def _evaluate_solution(
        solution: Solution,
        problem: Problem | None = None,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Evaluation:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"
        handler = CompositeHandler(
            handlers=[
                FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
                ArgdownParser(name="ArgdownParser"),
                HasArgdownHandler(),
                ArgMapCompositeHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=solution.argdown_snippet, source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation



class ArgMapFeedbackGenerator(FeedbackGenerator):
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
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"
        assert not evaluation.is_valid, (
            "Can only generate feedback for invalid solutions"
        )

        evaluation_issues = "\n".join(
            f"- **{k}**: {v}" for k, v in evaluation.metrics.items() if v
        )
        prompt = dedent("""
            Assignment: Give feedback and provide instructions for how to improve a given argument map.

            You will be shown an argument mapping problem, a student's preliminary solution, and its evaluation. Based on this information, provide feedback to the student and instructions for how to improve the solution.

                                                
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


class ConnectednessPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller number of weakly conncted components."""

    hints = [
        "In your map, only include arguments and claims that are dialectically connected (at least indirectly)."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown = evaluation.artifacts.get("argdown_map")
        assert argdown is not None and isinstance(argdown, ArgdownMultiDiGraph), (
            "Evaluation must contain a valid ArgdownMultiDiGraph artifact."
        )
        if argdown.number_of_nodes() == 0:
            return 0
        return len(list(nx.weakly_connected_components(argdown))) ** -1


class MaxArgsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of arguments."""

    hints = ["Include as many arguments as possible in your map."]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown = evaluation.artifacts.get("argdown_map")
        assert argdown is not None and isinstance(argdown, ArgdownMultiDiGraph), (
            "Evaluation must contain a valid ArgdownMultiDiGraph artifact."
        )

        return len(argdown.arguments)


class BalancePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with more balanced number of support and attack relations."""

    hints = ["Try to balance the number of support and attack relations in your map."]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        n_supp = sum(1 for dr in drs if dr.valence == Valence.SUPPORT)
        n_att = sum(1 for dr in drs if dr.valence == Valence.ATTACK)
        if n_supp + n_att == 0:
            return 0
        return 1 - abs(n_supp - n_att) / (n_supp + n_att)


class MaxSupportsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of support relations."""

    hints = ["Include as many support relations as possible in your map."]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        return sum(1 for dr in drs if dr.valence == Valence.SUPPORT)


class MaxAttacksPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of attack relations."""

    hints = ["Include as many attack relations as possible in your map."]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        return sum(1 for dr in drs if dr.valence == Valence.ATTACK)


class MaxDiameterPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger depth of the argument map."""

    hints = ["Try to create a 'deep' argument map with long chains of argumentation."]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        if argdown.number_of_nodes() == 0:
            return 0
        H = nx.DiGraph(argdown)
        if nx.is_directed_acyclic_graph(H):
            le = nx.dag_longest_path_length(H)
            #print(le)
        else:
            le = nx.diameter(H.to_undirected())
            #print("diameter:", le)
        return le


class MinDiameterPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller depth of the argument map."""

    hints = [
        "Try to create a 'shallow' argument map, where arguments and claims are directly related to the central claim(s)."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        if argdown.number_of_nodes() == 0:
            return 0
        H = nx.DiGraph(argdown)
        if nx.is_directed_acyclic_graph(H):
            le = nx.dag_longest_path_length(H)
            #print(le)
        else:
            le = nx.diameter(H.to_undirected())
            #print("diameter:", le)
        return 1 / (1 + le)


class DensityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger average degree of the argument map."""

    hints = [
        "Try to create a dense argument map with many dialectical relations between the identified arguments and claims."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        H = nx.DiGraph(argdown)
        degree_centrality = list(nx.degree_centrality(H).values())
        return (
            sum(degree_centrality) / len(degree_centrality) if degree_centrality else 0
        )


class MaxInDegreePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger maximum in degree of a node in the argument map."""

    hints = [
        "Try to create an argument map with a 'central' argument (or claim) that is supported or attacked by many other nodes."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        H = nx.DiGraph(argdown)
        in_degrees = list(dict(H.in_degree()).values())
        return max(in_degrees) if in_degrees else 0


class MaxOutDegreePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger maximum out degree of a node in the argument map."""

    hints = [
        "Try to create an argument map with a 'central' argument (or claim) which supports or attacks many other nodes."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        H = nx.DiGraph(argdown)
        out_degrees = list(dict(H.out_degree()).values())
        return max(out_degrees) if out_degrees else 0


class MinLeafsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller number of leaf nodes in the argument map."""

    hints = [
        "Try to create an argument map with a small _ratio_ of leaf nodes, i.e., of arguments and claims that are not supported or attacked by any other node."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        H = nx.DiGraph(argdown)
        leafs = [n for n in H.nodes if H.in_degree(n) == 0]
        return 1 - len(leafs) / H.number_of_nodes() if H.number_of_nodes() else 0


class ShortLabelsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with short labels."""

    hints = [
        "It's really important that your labels (for claims and arguments) are, on average, SHORT."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions
        ll = [len(a.label) if a.label else 0 for a in arguments] + [
            len(c.label) if c.label else 0 for c in claims
        ]
        return (round(sum(ll), -1) / len(ll)) ** -1 if ll else 0


class DiverseLabelsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with diverse labels."""

    hints = [
        "What really matters here is the diversity of your labels -- no two labels (of any argument or claim) should be alike."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions
        labels = [a.label for a in arguments] + [c.label for c in claims]
        lds = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1 = labels[i]
                l1 = l1 if l1 else ""
                l2 = labels[j]
                l2 = l2 if l2 else ""
                lds.append(textdistance.damerau_levenshtein.normalized_distance(l1, l2))

        return round(min(lds), 2) if lds else 0


class ShortClaimsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with succinct claims."""

    hints = [
        "Make sure that your claims are, on average, short and succinct. That's what counts at this point."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        claims: list[Proposition] = argdown.propositions
        ll = [len(c.texts[0]) if c.texts else 0 for c in claims]
        return round(sum(ll) / len(ll), -1) ** -1 if ll else 0


class LongClaimsPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with verbose claims."""

    hints = [
        "Make sure that your claims are, on average, long and verbose. That's what counts at this point."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        claims: list[Proposition] = argdown.propositions
        ll = [len(c.texts[0]) if c.texts else 0 for c in claims]
        return round(sum(ll) / len(ll), -1) if ll else 0


class ArgumentClaimSizePreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with arguments being on average 2-3 times as longs as claims."""

    hints = [
        "Make sure that your arguments' gists are neither too short nor too long; more specifically, they should be 2-3 times as long as the average claim in your map."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions
        cls = [len(c.texts[0]) if c.texts else 0 for c in claims]

        if not cls or not arguments:
            return 0

        mean_cl = sum(cls) / len(cls)
        good_args = [
            a
            for a in arguments
            if a.gists and 2 * mean_cl < len(a.gists[0]) < 3 * mean_cl
        ]

        return len(good_args) / len(arguments)


class IndependentWordingPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with independent wording of arguments and claims."""

    hints = [
        "Make sure that you render the arguments and claims *in your own words*, and independently from the formulations in the source text. This is crucial at this step."
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions

        dlds: list[float] = []
        for a in arguments:
            for g in a.gists:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_distance(
                        problem.sources, g
                    )
                )
        for c in claims:
            for t in c.texts:
                dlds.append(
                    textdistance.damerau_levenshtein.normalized_distance(
                        problem.sources, t
                    )
                )

        return round(sum(dlds) / len(dlds), 1) if dlds else 0


class SourceTextProximityPreferencePairGenerator(ScoringVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    that stick closely to the source text."""

    hints = [
        "Make sure that your argument map stays maximally faithful to and mimics closely the original source text!"
    ]

    def _score(
        self,
        problem: Problem,
        solution: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(solution, ArgumentMap), "Solution must be an ArgumentMap"
        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, solution.argdown_snippet
            ),
            1,
        )
