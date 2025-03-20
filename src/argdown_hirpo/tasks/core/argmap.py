from typing import Sequence

from abc import abstractmethod
import dataclasses
import random
from textwrap import dedent, shorten
import textdistance

import networkx as nx  # type: ignore
from pyargdown import (
    ArgdownEdge,
    ArgdownMultiDiGraph,
    Argument,
    Proposition,
    Valence,
    parse_argdown,
)

from argdown_hirpo.base import (
    Problem,
    Solution,
    Evaluation,
    Feedback,
    ChatPreferencePair,
    ProblemSolutionChat,
    ProblemGenerator,
    SolutionGenerator,
    Judge,
    FeedbackGenerator,
    VirtuePreferencePairGenerator,
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
        self, ask_for_invalid=False, hints: list[str] | None = None
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
            - indicate suppport and attack relations between nodes in accordance with Argdown syntax conventions;
            - do not include any detailed reconstructions of individual arguments as premise-conclusion-structures in your argdown code.

            Importantly, enclose your Argdown argument map in a single fenced codeblock, starting with '```argdown' and ending with '```'.                                                
        """)
            .strip()
            .format(sources=self.sources)
        )

        if ask_for_invalid:
            prompt += dedent("""\n\n
            > [!WARNING]
            > For didactic purposes, I want you to make mistakes in your answer.
            """)

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        return prompt

    def revise_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None
    ) -> str:
        prompt = "Revise your previously submitted argument map given the above evaluation and feedback."

        if ask_for_invalid:
            prompt += dedent("""\n\n
            > [!WARNING]
            > For didactic purposes, I still want you to make mistakes in your revised answer.
            """)

        if hints:
            prompt += "\n\nHints: " + " - ".join(hints)

        return prompt


@dataclasses.dataclass
class ArgumentMap(Solution):
    """Solution to the argument mapping problem: an argdown snippet."""

    argdown_snippet: str

    def __str__(self):
        return self.argdown_snippet


class ArgMapProblemGenerator(ProblemGenerator):
    async def arun(self, inputs) -> Problem:
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(i, str) for i in inputs)
        ):
            return ArgMapProblem(inputs)
        raise ValueError(
            "Inputs to an argument mapping problem must be a string or a list of strings"
        )


class ArgMapSolutionGenerator(SolutionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_solutions = kwargs.get("n_solutions", 10)
        self.temperature = kwargs.get("temperature", 0.5)
        self.max_tokens = kwargs.get("max_tokens", 2048)

    async def arun(
        self,
        problem: ArgMapProblem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[ArgumentMap]:
        assert isinstance(original_solution, ArgumentMap) or original_solution is None
        assert feedback or original_solution is None, (
            "Feedback is required for revised solutions"
        )

        messages = [
            {
                "role": "user",
                "content": problem.instruct_prompt(),
            }
        ]

        if original_solution and feedback:
            messages += [
                {
                    "role": "assistant",
                    "content": str(original_solution),
                },
                {
                    "role": "user",
                    "content": feedback.prompt,
                },
                {
                    "role": "assistant",
                    "content": feedback.feedback,
                },
                {
                    "role": "user",
                    "content": problem.revise_prompt(),
                },
            ]

        answers = await self._generate(
            messages,
            max_tokens=self.max_tokens,
            n=self.n_solutions,
            temperature=self.temperature,
        )

        argmaps: list[ArgumentMap] = []

        # postprocess: extract fenced code block
        for answer in answers:
            if answer.count("```argdown") == 1:
                if answer.split("```argdown")[1].count("\n```") == 1:
                    answer = answer.split("```argdown")[1].split("\n```")[0]
                    answer = "```argdown" + answer + "\n```"
            argmaps.append(ArgumentMap(argdown_snippet=answer))

        return argmaps


class ArgMapJudge(Judge):
    """Judge for the argument mapping task."""

    def _evaluate_argmap(
        self, problem: ArgMapProblem, argmap: ArgumentMap
    ) -> Evaluation:
        is_valid = True
        eval_data = {
            "fenced_code_block": "",
            "invalid_argdown_syntax": "",
            "missing_labels": "",
            "duplicate_labels": "",
            "premise_conclusion_structures": "",
        }

        ads = argmap.argdown_snippet.strip("\n ")
        if ads.startswith("```argdown") and ads.endswith("```"):
            ads = "\n".join(ads.splitlines()[1:-1])
        else:  # no fenced code block
            is_valid = False
            error_msg = "Failed to extract single fenced argdown block:"
            if ads.count("```argdown") == 0:
                error_msg += " No fenced code block starting with '```argdown'."
            if ads.count("```argdown") > 1:
                error_msg += (
                    " More than one fenced code block starting with '```argdown'."
                )
            if "```\n" not in ads:
                error_msg += " No closing '```'."
            eval_data["fenced_code_block"] = error_msg

        try:
            argdown = parse_argdown(ads)
        except Exception as e:
            argdown = None
            is_valid = False
            eval_data["invalid_argdown_syntax"] = f"Failed to parse argdown: {str(e)}"

        if argdown:
            incomplete_claims: list[str] = []
            for claim in argdown.propositions:
                assert isinstance(claim, Proposition)
                if claim.label is None or "UNNAMED_PROPOSITION" in claim.label:
                    if not claim.texts or not claim.texts[0]:
                        incomplete_claims.append("Empty claim")
                    else:
                        incomplete_claims.append(shorten(claim.texts[0], width=40))
            if incomplete_claims:
                is_valid = False
                eval_data["missing_claim_labels"] = (
                    f"Missing labels for nodes: {', '.join(incomplete_claims)}"
                )

            labels = [a.label for a in argdown.arguments if a.label] + [
                c.label for c in argdown.propositions if c.label
            ]
            duplicates = set([l for l in labels if labels.count(l) > 1])
            if duplicates:
                is_valid = False
                eval_data["duplicate_labels"] = (
                    f"Duplicate labels: {', '.join(duplicates)}"
                )

            for argument in argdown.arguments:
                assert isinstance(argument, Argument)
                if argument.pcs:
                    is_valid = False
                    eval_data["premise_conclusion_structures"] = (
                        f"Found detailed reconstruction of individual argument <{argument.label}> as premise-conclusion-structures."
                    )
                    break

        return Evaluation(
            is_valid=is_valid, artifacts={"argdown": argdown, "eval_metrics": eval_data}
        )

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
        self.n_feedbacks = kwargs.get("n_solutions", 5)
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
            f"- **{k}**: {v}"
            for k, v in evaluation.artifacts.get("eval_metrics", {}).items()
            if v
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

        return [Feedback(feedback=answer, prompt=prompt) for answer in answers]


class ArgMapVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the argument mapping task."""

    hints: list[str] = []

    @abstractmethod
    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        pass

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert isinstance(problem, ArgMapProblem), "Problem must be an ArgMapProblem"
        assert all(isinstance(s, ArgumentMap) for s in candidate_solutions), (
            "All solutions must be ArgumentMaps"
        )
        assert original_solution is None or isinstance(
            original_solution, ArgumentMap
        ), "Original solution must be an ArgumentMap"
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )

        pairs: list[ChatPreferencePair] = []

        # rank valid argmaps according to the _score function
        valid_argmaps: list[tuple[ArgumentMap, Evaluation]] = list(
            zip(candidate_solutions, evaluations)  # type: ignore
        )
        valid_argmaps.sort(key=lambda x: self._score(problem, x[0], x[1]), reverse=True)
        valid_argmaps = [
            (solution, evaluation)
            for solution, evaluation in valid_argmaps
            if evaluation.is_valid
            and evaluation.artifacts["argdown"].number_of_nodes() > 1
        ]

        if len(valid_argmaps) < 2:
            return pairs
        top_score = self._score(problem, *valid_argmaps[0])
        if top_score == self._score(problem, *valid_argmaps[-1]):
            return pairs

        top_argmap, _ = valid_argmaps[0]
        weaker_argmap = random.choice(
            [s for s, e in valid_argmaps if self._score(problem, s, e) < top_score]
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_argmap,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=weaker_argmap,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
            )
        )

        return pairs


class ConnectednessPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller number of weakly conncted components."""

    hints = [
        "In your map, only include arguments and claims that are dialectically connected (at least indirectly)."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        if argdown.number_of_nodes() == 0:
            return 0
        return len(list(nx.weakly_connected_components(argdown))) ** -1


class MaxArgsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of arguments."""

    hints = ["Include as many arguments as possible in your map."]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        return len(argdown.arguments)


class BalancePreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with more balanced number of support and attack relations."""

    hints = ["Try to balance the number of support and attack relations in your map."]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        n_supp = sum(1 for dr in drs if dr.valence == Valence.SUPPORT)
        n_att = sum(1 for dr in drs if dr.valence == Valence.ATTACK)
        if n_supp + n_att == 0:
            return 0
        return 1 - abs(n_supp - n_att) / (n_supp + n_att)


class MaxSupportsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of support relations."""

    hints = ["Include as many support relations as possible in your map."]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        return sum(1 for dr in drs if dr.valence == Valence.SUPPORT)


class MaxAttacksPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger number of attack relations."""

    hints = ["Include as many attack relations as possible in your map."]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        drs: list[ArgdownEdge] = argdown.dialectical_relations
        return sum(1 for dr in drs if dr.valence == Valence.ATTACK)


class MaxDiameterPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger depth of the argument map."""

    hints = ["Try to create a 'deep' argument map with long chains of argumentation."]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        return nx.diameter(argdown) if argdown.number_of_nodes() > 1 else 0


class MinDiameterPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller depth of the argument map."""

    hints = [
        "Try to create a 'shallow' argument map, where arguments and claims are directly related to the central claim(s)."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        return (1 + nx.diameter(argdown)) ** -1 if argdown.number_of_nodes() > 1 else 0


class DensityPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger average degree of the argument map."""

    hints = [
        "Try to create a dense argument map with many dialectical relations between the identified arguments and claims."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        degree_centrality = list(nx.degree_centrality(argdown).values())
        return (
            sum(degree_centrality) / len(degree_centrality) if degree_centrality else 0
        )


class MaxInDegreePreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger maximum in degree of a node in the argument map."""

    hints = [
        "Try to create an argument map with a 'central' argument (or claim) that is supported or attacked by many other nodes."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        in_degrees = list(dict(argdown.in_degree()).values())
        return max(in_degrees) if in_degrees else 0


class MaxOutDegreePreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with larger maximum out degree of a node in the argument map."""

    hints = [
        "Try to create an argument map with a 'central' argument (or claim) which supports or attacks many other nodes."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        out_degrees = list(dict(argdown.out_degree()).values())
        return max(out_degrees) if out_degrees else 0


class MinLeafsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with smaller number of leaf nodes in the argument map."""

    hints = [
        "Try to create an argument map with a small _ratio_ of leaf nodes, i.e., of arguments and claims that are not supported or attacked by any other node."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        leafs = [n for n in argdown.nodes if argdown.out_degree(n) == 0]
        return (
            1 - len(leafs) / argdown.number_of_nodes()
            if argdown.number_of_nodes()
            else 0
        )


class ShortLabelsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with short labels."""

    hints = [
        "It's really important that your labels (for claims and arguments) are, on average, SHORT."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions
        ll = [len(a.label) if a.label else 0 for a in arguments] + [
            len(c.label) if c.label else 0 for c in claims
        ]
        return sum(ll) / len(ll) if ll else 0


class DiverseLabelsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with diverse labels."""

    hints = [
        "What really matters here is the diversity of your labels -- no two labels (of any argument or claim) should be alike."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
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

        return min(lds) if lds else 0


class ShortClaimsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with succinct claims."""

    hints = [
        "Make sure that your claims are, on average, short and succinct. That's what counts at this point."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        claims: list[Proposition] = argdown.propositions
        ll = [len(c.label) if c.label else 0 for c in claims]
        return 1 / (1 + sum(ll) / len(ll)) if ll else 0


class LongClaimsPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with verbose claims."""

    hints = [
        "Make sure that your claims are, on average, long and verbose. That's what counts at this point."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        claims: list[Proposition] = argdown.propositions
        ll = [len(c.label) if c.label else 0 for c in claims]
        return sum(ll) / len(ll) if ll else 0


class ArgumentClaimSizePreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with arguments being on average 2-3 times as longs as claims."""

    hints = [
        "Make sure that your arguments' gists are neither too short nor too long; more specifically, they should be 2-3 times as long as the average claim in your map."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
        arguments: list[Argument] = argdown.arguments
        claims: list[Proposition] = argdown.propositions
        cls = [len(c.label) if c.label else 0 for c in claims]

        if not cls or not arguments:
            return 0

        mean_cl = sum(cls) / len(cls)
        good_args = [
            a for a in arguments if a.label and 2 * mean_cl < len(a.label) < 3 * mean_cl
        ]

        return len(good_args) / len(arguments)


class IndependentWordingPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    with independent wording of arguments and claims."""

    hints = [
        "Make sure that you render the arguments and claims *in your own words*, and independently from the formulations in the source text. This is crucial at this step."
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        argdown: ArgdownMultiDiGraph = evaluation.artifacts["argdown"]
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

        return sum(dlds) / len(dlds) if dlds else 0


class SourceTextProximityPreferencePairGenerator(ArgMapVirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the annotation task, prefering valid argument maps
    that stick closely to the source text."""

    hints = [
        "Make sure that your argument map stays maximally faithful to and mimics closely the original source text!"
    ]

    def _score(
        self,
        problem: ArgMapProblem,
        argmap: ArgumentMap,
        evaluation: Evaluation,
    ) -> float:
        return textdistance.damerau_levenshtein.normalized_similarity(
            problem.sources, argmap.argdown_snippet
        )
