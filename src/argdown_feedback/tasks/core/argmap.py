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
    Problem,
    ScoringVirtuePreferencePairGenerator,
    Solution,
    Evaluation,
    Feedback,
    ProblemGenerator,
    Judge,
    FeedbackGenerator,
)
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.core.argmap_handler import ArgMapCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import HasArgdownHandler
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
    FencedCodeBlockExtractor,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationDType,
    VerificationRequest,
)


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
            Assignment: Reconstruct a source text's argumentation as an informal Argdown argument map.
                        
            Analyse the argumentation in the following source text by creating an Argdown argument map.

            ::: {{.source_text}}              
            {sources}
            :::

            In particular, I ask you to

            - explicitly label all nodes in the argument map;
            - use square/angled brackets for labels to distinguish arguments/claims;
            - indicate support and attack relations between nodes in accordance with Argdown syntax conventions;
            - do not include any detailed reconstructions of individual arguments as premise-conclusion-structures in your argdown code.

            Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.                                                
        """)
            .strip()
            .format(sources=self.sources)
        )

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        if ask_for_invalid:
            prompt += (
                "\n\n"
                "> [!WARNING]\n"
                "> For didactic purposes, I want you to make mistakes in your answer.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"

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
            prompt += (
                "\n\n"
                "> [!WARNING]\n"
                "> For didactic purposes, I still want you to make mistakes in your revised answer.\n"
            )

            if evaluation:
                metrics = {k: v for k, v in evaluation.metrics.items() if v}
                if metrics:
                    prompt += "> Expected errors:\n"
                    for k, v in metrics.items():
                        prompt += f"> - {k}: {v}\n"

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


class ArgMapJudge(Judge):
    """Judge for the argument mapping task."""

    def _evaluate_argmap(
        self, problem: ArgMapProblem, argmap: ArgumentMap
    ) -> Evaluation:
        handler = CompositeHandler(
            handlers=[
                DefaultProcessingHandler(),
                HasArgdownHandler(),
                ArgMapCompositeHandler(),
            ]
        )
        request = VerificationRequest(
            inputs=argmap.argdown_snippet, source=problem.sources
        )
        result = handler.process(request)
        evaluation = Evaluation.from_verification_request(result)
        if evaluation.artifacts.get("argdown_map") is None:
            evaluation.artifacts["argdown_map"] = evaluation.artifacts.get("argdown")
        return evaluation

    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(original_solution, ArgumentMap) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for evaluating revised solutions"
        )

        evaluations = []
        for solution in solutions:
            assert isinstance(solution, ArgumentMap), (
                "All solutions must be ArgumentMaps"
            )
            evaluations.append(self._evaluate_argmap(problem, solution))

        return evaluations


class ArgMapFeedbackGenerator(FeedbackGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_feedbacks = kwargs.get("n_feedbacks", 5)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

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
            max_tokens=self.max_tokens,
            n=self.n_feedbacks,
            temperature=self.temperature,
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
        argmap: Solution,
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
        argmap: Solution,
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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
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
        argmap: Solution,
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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        if argdown.number_of_nodes() == 0:
            return 0
        H = nx.DiGraph(argdown)
        if nx.is_directed_acyclic_graph(H):
            le = nx.dag_longest_path_length(H)
            print(le)
        else:
            le = nx.diameter(H.to_undirected())
            print("diameter:", le)
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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown_map"]
        if argdown.number_of_nodes() == 0:
            return 0
        H = nx.DiGraph(argdown)
        if nx.is_directed_acyclic_graph(H):
            le = nx.dag_longest_path_length(H)
            print(le)
        else:
            le = nx.diameter(H.to_undirected())
            print("diameter:", le)
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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
                lds.append(textdistance.levenshtein.normalized_distance(l1, l2))

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"

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
        argmap: Solution,
        evaluation: Evaluation,
    ) -> float:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert isinstance(argmap, ArgumentMap), "Solution must be an ArgumentMap"
        return round(
            textdistance.damerau_levenshtein.normalized_similarity(
                problem.sources, argmap.argdown_snippet
            ),
            1,
        )
