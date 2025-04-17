"Base HIR preference pair generators."

from abc import ABC, abstractmethod
import dataclasses
import logging
import random
from statistics import mean
from textwrap import dedent
from typing import Any, Sequence, TypedDict

from bs4 import BeautifulSoup
from openai import AsyncOpenAI, OpenAI
from pyargdown import Argdown
import tenacity

from argdown_feedback.verifiers.verification_request import VerificationDType, VerificationRequest

logger = logging.getLogger(__name__)


# DATA MODELS
######################


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatPreferencePair(TypedDict):
    chosen: list[ChatMessage]
    rejected: list[ChatMessage]


class Problem(ABC):
    """Abstract base class representing a problem."""

    @abstractmethod
    def instruct_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation = None
    ) -> str:
        """Cast the problem as a problem statement, including an instruction to solve it."""

    @abstractmethod
    def revise_prompt(
        self, ask_for_invalid=False, hints: list[str] | None = None, evaluation = None
    ) -> str:
        """Instruction to revise earlier mentioned solution of the problem."""


class Solution(ABC):
    """Abstract base class representing a solution."""

    @abstractmethod
    def __str__(self) -> str:
        """Cast the solution as a string."""

    @classmethod
    @abstractmethod
    def from_raw_answer(cls, answer: str) -> "Solution":
        """Cast a raw answer as a solution."""



@dataclasses.dataclass
class Evaluation:
    """
    Evaluation of a solution

    Every solution is valid or invalid. -- Which can mean different things
    depending on the problem, and the criteria used to evaluate it.

    The ability to generate *valid* solutions -- in which ever way validity
    is interpreted -- is the primary skill HIRPO seeks to instill in LLMs.

    On top of marking a solution as valid or invalid, the evaluation may
    contain additional information about the solution, or evaluation artifacts.
    Such additional information, artifacts or metrics may be used by the
    feedback generator or by the virtue preference pair generator.
    """

    is_valid: bool
    artifacts: dict[str, Any]  # global artifacts
    metrics: dict[str, Any]

    @classmethod
    def from_verification_request(
        cls,
        request: VerificationRequest,
    ) -> "Evaluation":
        """
        Create an Evaluation from a VerificationRequest.
        """

        metrics = dict((f"{e+1:02d}_{result.verifier_id}", result.message) for e, result in enumerate(request.results))
        
        artifacts = {}

        last_soup = next(
            (
                data for data in reversed(request.verification_data)
                if data.dtype == VerificationDType.xml and data.data is not None
                and isinstance(data.data, BeautifulSoup)
            ),
            None,
        )
        artifacts["soup"] = last_soup.data if last_soup is not None else None

        verification_data_argdown = [
            data for data in request.verification_data
            if data.dtype == VerificationDType.argdown and data.data is not None
            and isinstance(data.data, Argdown)
        ]

        last_argdown = next(reversed(verification_data_argdown), None)
        artifacts["argdown"] = last_argdown.data if last_argdown is not None else None

        # argdown_map
        last_argdown_map = next(
            (
                data for data in reversed(verification_data_argdown)
                if data.metadata and data.metadata.get("filename")=="map.ad"
            ),
            None,
        )
        artifacts["argdown_map"] = last_argdown_map.data if last_argdown_map is not None else None

        # argdown_reco        
        last_argdown_reco = next(
            (
                data for data in reversed(verification_data_argdown)
                if data.metadata and data.metadata.get("filename")=="reconstructions.ad"
            ),
            None,
        )
        artifacts["argdown_reco"] = last_argdown_reco.data if last_argdown_reco is not None else None

        # formalizations are stored as details in result of WellFormedFormulasHandler
        all_expressions = None
        all_declarations = None
        if last_argdown is not None:
            argdown_vd_id = last_argdown_reco.id if last_argdown_reco is not None else last_argdown.id
            wff_result = next(
                (
                    result for result in request.results
                    if result.verifier_id == "WellFormedFormulasHandler" and result.verification_data_references==[argdown_vd_id]),
                None,
            )
            if wff_result is not None:
                all_expressions = wff_result.details.get("all_expressions")
                all_declarations = wff_result.details.get("all_declarations")
        artifacts["all_expressions"] = all_expressions
        artifacts["all_declarations"] = all_declarations

        return cls(
            is_valid=request.is_valid(),
            artifacts=artifacts,
            metrics=metrics,
        )


@dataclasses.dataclass
class Feedback:
    prompt: str
    feedback: str


@dataclasses.dataclass
class ProblemSolutionChat:
    """
    Dataclass representing a problem-solving chat.
    Starts with the problem presentation, and concludes with the final solution.
    Possibly contains a revision phase (original solution and feedback) inbetween.
    If original_solution and feedback are not None, solution represents the final,
    revised solution.
    """

    problem: Problem
    solution: Solution
    original_solution: Solution | None = None
    feedback: Feedback | None = None

    def as_chat(self, **problem_kwargs) -> list[ChatMessage]:
        if self.original_solution is None or self.feedback is None:
            return [
                ChatMessage(
                    role="user", content=self.problem.instruct_prompt(**problem_kwargs)
                ),
                ChatMessage(role="assistant", content=str(self.solution)),
            ]
        return [
            ChatMessage(role="user", content=self.problem.instruct_prompt()),
            ChatMessage(role="assistant", content=str(self.original_solution)),
            ChatMessage(role="user", content=self.feedback.prompt),
            ChatMessage(role="assistant", content=self.feedback.feedback),
            ChatMessage(
                role="user", content=self.problem.revise_prompt(**problem_kwargs)
            ),
            ChatMessage(role="assistant", content=str(self.solution)),
        ]


# GENERATORS
######################


class HIRAbstractGenerator(ABC):
    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        pass


class HIRAbstractGeneratorLLM(ABC):

    def __init__(self, *args, **kwargs):
        inference_base_url = kwargs.get("inference_base_url", None)
        model_id = kwargs.get("model_id", None)
        if inference_base_url and model_id:
            self.model_id = model_id
            self.client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=inference_base_url,
            )
            models = OpenAI(api_key="EMPTY", base_url=inference_base_url).models.list()
            if self.model_id not in [d.id for d in models.data]:
                logging.getLogger(__name__).warning(
                    f"Model {self.model_id} not found in OpenAI API. Model served: {models}. Will proceed with first of these models."
                )
                self.model_id = next(models.data[0].id)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=60),
        stop=tenacity.stop_after_attempt(6),
    )
    async def _generate(self, messages, **gen_kwargs):
        stream = False
        completion = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=stream,
            **gen_kwargs,
        )
        answers = [choice.message.content for choice in completion.choices]
        answers = [a for a in answers if a is not None]
        return answers


class ProblemGenerator(HIRAbstractGenerator):
    """Generates a problem."""

    @abstractmethod
    async def arun(self, inputs) -> Problem:
        pass


class ProblemGeneratorLLM(HIRAbstractGeneratorLLM, ProblemGenerator):
    """Generates a problem."""



class SolutionGenerator(HIRAbstractGeneratorLLM):
    """Generates solutions."""

    @abstractmethod
    async def arun(
        self,
        Problem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Solution]:
        pass


class GenericSolutionGenerator(SolutionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # solution class
        self.solution_class = kwargs.get("solution_class", Solution)
        assert self.solution_class, "Solution class is required"
        assert issubclass(
            self.solution_class, Solution
        ), "Solution class must be a subclass of Solution"
        # generation kwargs
        self.n_solutions = kwargs.get("n_solutions", 10)
        self.temperature = kwargs.get("temperature", 0.5)
        self.max_tokens = kwargs.get("max_tokens", 2048)

    async def arun(
        self,
        problem: Problem,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Solution]:
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

        # remove empty and duplicate answers
        answers = [a for a in answers if a]
        answers = list(set(answers))

        recos: list[Solution] = []

        for answer in answers:
            reco = self.solution_class.from_raw_answer(answer)
            recos.append(reco)  # type: ignore

        return recos

class Judge(HIRAbstractGenerator):
    """Judges solutions."""

    @abstractmethod
    async def arun(
        self,
        problem: Problem,
        solutions: Sequence[Solution],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> Sequence[Evaluation]:
        pass


# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#     BIG REFACTORING TODO     # 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #


class BaseJudge(HIRAbstractGenerator):
    """
    BaseJudge implenents skeleton handler / chain-of-repsonsibility pattern
    Specific judges individually construct handler chains fromVerifiers
    """



class FeedbackGenerator(HIRAbstractGeneratorLLM):
    """Generates feedback."""

    @abstractmethod
    async def arun(
        self, problem: Problem, solution: Solution, evaluation: Evaluation
    ) -> Sequence[Feedback]:
        pass



class GenericFeedbackGenerator(FeedbackGenerator):
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





class VirtuePreferencePairGenerator(HIRAbstractGenerator):
    """
    Generates preference pairs from differences in terms of additional
    virtues other than (syntactic) validity of candidate_solutions.
    """

    @abstractmethod
    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pass


                        
class ScoringVirtuePreferencePairGenerator(VirtuePreferencePairGenerator):
    """Generate virtue-preference pairs for the informal argument reconstruction task
    based on score."""

    hints: list[str] = []

    @abstractmethod
    def _score(
        self,
        problem: Problem,
        reco: Solution,
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
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )

        pairs: list[ChatPreferencePair] = []

        # rank valid recos according to the _score function
        valid_recos: list[tuple[Solution, Evaluation]] = list(
            zip(candidate_solutions, evaluations)  # type: ignore
        )
        valid_recos.sort(key=lambda x: self._score(problem, x[0], x[1]), reverse=True)
        valid_recos = [
            (solution, evaluation)
            for solution, evaluation in valid_recos
            if evaluation.is_valid
        ]

        if len(valid_recos) < 2:
            return pairs
        top_score = self._score(problem, *valid_recos[0])
        if top_score == self._score(problem, *valid_recos[-1]):
            return pairs

        top_reco, _ = valid_recos[0]
        weaker_reco = random.choice(
            [s for s, e in valid_recos if self._score(problem, s, e) < top_score]
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=weaker_reco,
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=self.hints),
            )
        )

        return pairs




class FailureTypePreferencePairGenerator(HIRAbstractGenerator):
    """
    Generates preference pairs from failure type differences
    between candidate solutions.
    """

    @abstractmethod
    async def arun(
        self,
        problem,
        candidate_solutions,
        evaluations,
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        pass



class GenericFailureDiffPreferencePairGenerator(FailureTypePreferencePairGenerator):
    """Generate failure-type-preference pairs based on the differences in failure profiles."""

    avoid_errors_hint = (
        "Very important! No matter whether your solution fully satisfies the above instructions or not, "
        "make sure to avoid the following mistakes:\n"
    )

    async def arun(
        self,
        problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Number of solutions must match number of evaluations"
        )
        pairs: list[ChatPreferencePair] = []

        # count error types
        error_counts: dict[str, int] = {}
        for key in evaluations[0].metrics.keys():
            error_counts[key] = len([1 for e in evaluations if e.metrics[key]])
        # dismiss errors that are never avoided, i.e. always present
        error_counts = {k: v for k, v in error_counts.items() if 0 < v < len(evaluations)}

        if not error_counts:
            return pairs

        # get error type that is most common
        most_common_type, _ = max(
            error_counts.items(),
            key=lambda x: x[1],
        )

        # chosen solution avoids most common error
        chosen_idx = random.choice(
            [
                idx
                for idx, eval in enumerate(evaluations)
                if not eval.metrics[most_common_type]
            ]
        )
        # rejected solution commits most common error
        rejected_idx = random.choice(
            [
                idx
                for idx, eval in enumerate(evaluations)
                if eval.metrics[most_common_type]
            ]
        )

        # list all errors avoided by the chosen solution
        errors_avoided = {
            k: v for k, v in evaluations[rejected_idx].metrics.items()
            if v and not evaluations[chosen_idx].metrics[k]
        }

        hint = (
            self.avoid_errors_hint
            + "\n".join(f"- {k}: {v}" for k, v in errors_avoided.items())
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=candidate_solutions[chosen_idx],
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=[hint]),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=candidate_solutions[rejected_idx],
                    feedback=feedback,
                    original_solution=original_solution,
                ).as_chat(hints=[hint]),
            )
        )

        return pairs


# MAIN GENERATORS
######################


class HIRPreferencePairGenerator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        virtue_preference_pair_generator: VirtuePreferencePairGenerator | list[VirtuePreferencePairGenerator] | None = None,
        feedback_generator: FeedbackGenerator | None = None,
        failure_type_preference_pair_generator: FailureTypePreferencePairGenerator
        | None = None,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        self.feedback_generator = feedback_generator
        if virtue_preference_pair_generator is None:
            virtue_preference_pair_generator = []
        if isinstance(virtue_preference_pair_generator, VirtuePreferencePairGenerator):
            virtue_preference_pair_generator = [virtue_preference_pair_generator]
        self.virtue_preference_pair_generators = virtue_preference_pair_generator
        self.failure_type_preference_pair_generator = (
            failure_type_preference_pair_generator
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validity_vs_virtue_router(
        self, mean_syntactic_validity: float
    ) -> tuple[bool, bool]:
        do_virtue_hirp = (
            bool(self.virtue_preference_pair_generators)
            and random.random() < mean_syntactic_validity
        )
        do_validity_hirp = not do_virtue_hirp
        return do_validity_hirp, do_virtue_hirp

    def build_validity_pref_pair(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        """
        Builds two preference pairs based on syntactic differences
        between candidate_solutions, varies the instruction in
        accordance with HIR.
        """
        pairs: list[ChatPreferencePair] = []

        top_valid_solution = next(
            cs for cs, e in zip(candidate_solutions, evaluations) if e.is_valid
        )
        top_invalid_solution = next(
            cs for cs, e in zip(candidate_solutions, evaluations) if not e.is_valid
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(),
            )
        )

        pairs.append(
            ChatPreferencePair(
                chosen=ProblemSolutionChat(
                    problem=problem,
                    solution=top_invalid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True),
                rejected=ProblemSolutionChat(
                    problem=problem,
                    solution=top_valid_solution,
                    original_solution=original_solution,
                    feedback=feedback,
                ).as_chat(ask_for_invalid=True),
            )
        )

        return pairs

    async def build_solution_pref_pair(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
        original_solution: Solution | None = None,
        feedback: Feedback | None = None,
    ) -> list[ChatPreferencePair]:
        assert len(candidate_solutions) == len(evaluations), (
            "Candidate solutions and evaluations must have the same length."
        )
        pairs: list[ChatPreferencePair] = []
        mean_syntactic_validity = mean(int(e.is_valid) for e in evaluations)
        do_validity_hirp, do_virtue_hirp = self.validity_vs_virtue_router(
            mean_syntactic_validity
        )
        if do_virtue_hirp:
            logger.debug("Constructing virtue preference pair")
            assert self.virtue_preference_pair_generators, "Internal error: Attempting do_virtue_hirp while no virtue preference pair generators available."
            virtue_preference_pair_generator = random.choice(self.virtue_preference_pair_generators)
            pairs.extend(
                await virtue_preference_pair_generator.arun(
                    problem,
                    candidate_solutions,
                    evaluations,
                    original_solution=original_solution,
                    feedback=feedback,
                )
            )
        if do_validity_hirp:
            logger.debug("Constructing syntactic validity preference pair")
            pairs.extend(
                self.build_validity_pref_pair(
                    problem,
                    candidate_solutions,
                    evaluations,
                    original_solution=original_solution,
                    feedback=feedback,
                )
            )
        return pairs

    async def run_self_critique(
        self,
        problem: Problem,
        candidate_solutions: Sequence[Solution],
        evaluations: Sequence[Evaluation],
    ) -> list[ChatPreferencePair]:
        pairs: list[ChatPreferencePair] = []
        if self.feedback_generator is None:
            return pairs

        for cs, e in zip(candidate_solutions, evaluations):
            if e.is_valid:
                continue
            feedbacks = await self.feedback_generator.arun(problem, cs, e)
            revisions: list[Sequence[Solution]] = []
            revision_evaluations: list[Sequence[Evaluation]] = []
            for feedback in feedbacks:
                candidate_revisions = await self.solution_generator.arun(
                    problem, original_solution=cs, feedback=feedback
                )
                if not candidate_revisions:
                    continue
                revision_evals = await self.judge.arun(
                    problem,
                    candidate_revisions,
                    original_solution=cs,
                    feedback=feedback,
                )
                revisions.append(candidate_revisions)
                revision_evaluations.append(revision_evals)

                # run revision-specific HIRP to generate solution preference pairs

                if not any(re.is_valid for re in revision_evals):
                    if self.failure_type_preference_pair_generator is not None:
                        pairs.extend(
                            await self.failure_type_preference_pair_generator.arun(
                                problem, candidate_solutions, evaluations
                            )
                        )
                    continue

                if all(re.is_valid for re in revision_evals):
                    pairs.extend(
                        await self.build_solution_pref_pair(
                            problem,
                            candidate_solutions=[cs, candidate_revisions[0]],
                            evaluations=[e, revision_evals[0]],
                        )
                    )

                pairs.extend(
                    await self.build_solution_pref_pair(
                        problem,
                        candidate_revisions,
                        revision_evals,
                        original_solution=cs,
                        feedback=feedback,
                    )
                )

            # generate feedback preference pair

            # get most and least successful feedback by counting valid revisions
            n_valid_revisions = [
                sum(e.is_valid for e in re) for re in revision_evaluations
            ]
            if n_valid_revisions and max(n_valid_revisions) > min(n_valid_revisions):
                most_successful_feedback = feedbacks[
                    n_valid_revisions.index(max(n_valid_revisions))
                ]
                least_successful_feedback = feedbacks[
                    n_valid_revisions.index(min(n_valid_revisions))
                ]
                pairs.append(
                    ChatPreferencePair(
                        chosen=[
                            ChatMessage(role="user", content=problem.instruct_prompt()),
                            ChatMessage(role="assistant", content=str(cs)),
                            ChatMessage(
                                role="user", content=most_successful_feedback.prompt
                            ),
                            ChatMessage(
                                role="assistant",
                                content=most_successful_feedback.feedback,
                            ),
                        ],
                        rejected=[
                            ChatMessage(role="user", content=problem.instruct_prompt()),
                            ChatMessage(role="assistant", content=str(cs)),
                            ChatMessage(
                                role="user", content=most_successful_feedback.prompt
                            ),
                            ChatMessage(
                                role="assistant",
                                content=least_successful_feedback.feedback,
                            ),
                        ],
                    )
                )

        return pairs

    def self_critique_router(self, evaluations: Sequence[Evaluation]) -> tuple[bool, bool]:
        """router for main workflow"""
        do_self_critique = not any(e.is_valid for e in evaluations)
        return do_self_critique, not do_self_critique

    async def arun(self, inputs) -> list[ChatPreferencePair]:
        """main workflow logic"""
        problem = await self.problem_generator.arun(inputs)
        candidate_solutions = await self.solution_generator.arun(problem)
        evaluations = await self.judge.arun(problem, candidate_solutions)

        pairs: list[ChatPreferencePair] = []

        do_self_critique, skip_self_critique = self.self_critique_router(evaluations)

        if do_self_critique:
            pairs = await self.run_self_critique(
                problem, candidate_solutions, evaluations
            )

        if skip_self_critique:
            pairs = await self.build_solution_pref_pair(
                problem, candidate_solutions, evaluations
            )

        if not pairs and self.failure_type_preference_pair_generator is not None:
            pairs = await self.failure_type_preference_pair_generator.arun(
                problem, candidate_solutions, evaluations
            )

        return pairs



class HIREvaluator(HIRAbstractGenerator):
    def __init__(
        self,
        problem_generator: ProblemGenerator,
        solution_generator: SolutionGenerator,
        judge: Judge,
        **kwargs,
    ):
        self.problem_generator = problem_generator
        self.solution_generator = solution_generator
        self.judge = judge
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def arun(self, inputs) -> float:
        """evaluate inputs and returns average accuracy of the candidate solutions"""
        problem = await self.problem_generator.arun(inputs)
        candidate_solutions = await self.solution_generator.arun(problem)
        evaluations = await self.judge.arun(problem, candidate_solutions)
        accuracy = sum(int(e.is_valid) for e in evaluations) / len(evaluations)
        return accuracy