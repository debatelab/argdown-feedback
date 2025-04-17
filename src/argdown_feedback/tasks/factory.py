# Factory methods for creating HITPO instances from configs

import logging
from typing import Any

from argdown_feedback.tasks import registry
from argdown_feedback.tasks.base import GenericFailureDiffPreferencePairGenerator, GenericSolutionGenerator, HIRAbstractGenerator, HIRAbstractGeneratorLLM, HIREvaluator, HIRPreferencePairGenerator



logger = logging.getLogger(__name__)

class HIRPOFactory:

    @staticmethod
    def hirpo_gen_from_config(
        solution: str,
        problem_generator: str,
        judge: str,
        model_kwargs: dict[str, Any],
        feedback_generator: str | None = None,
        virtue_preference_pair_generator: str | list[str] | None = None,
        failure_type_preference_pair_generator: str | None = None,
        **kwargs,
    ) -> HIRPreferencePairGenerator | None:
        """Create a HIRPreferencePairGenerator from config."""
        try:

            pg_class = registry.get_class(problem_generator)
            if issubclass(pg_class, HIRAbstractGeneratorLLM):
                pg_instance = pg_class(**model_kwargs)
            elif issubclass(pg_class, HIRAbstractGenerator):
                pg_instance = pg_class()
            else:
                raise TypeError(f"Invalid problem generator class: {pg_class} of type {type(pg_class)}")

            sg_instance = GenericSolutionGenerator(
                solution_class=registry.get_class(solution),
                **model_kwargs,
            )

            jg_class = registry.get_class(judge)
            if issubclass(jg_class, HIRAbstractGeneratorLLM):
                jg_instance = jg_class(**model_kwargs)
            elif issubclass(jg_class, HIRAbstractGenerator):
                jg_instance = jg_class()
            else:
                raise TypeError(f"Invalid judge class: {jg_class} of type {type(jg_class)}")

            if feedback_generator is None:
                fg_instance = None
            else:
                fg_class = registry.get_class(feedback_generator)
                if issubclass(fg_class, HIRAbstractGeneratorLLM):
                    fg_instance = fg_class(**model_kwargs)
                elif issubclass(fg_class, HIRAbstractGenerator):
                    fg_instance = fg_class()
                else:
                    raise TypeError(f"Invalid feedback generator class: {fg_class} of type {type(fg_class)}")

            if virtue_preference_pair_generator is None:
                virtue_preference_pair_generator = []
            if isinstance(virtue_preference_pair_generator, str):
                virtue_preference_pair_generator = [virtue_preference_pair_generator]
            vppg_instances = [
                registry.get_class(v)() for v in virtue_preference_pair_generator
            ]
            if failure_type_preference_pair_generator is not None:
                ftppg_instance = registry.get_class(failure_type_preference_pair_generator)()
            else:
                ftppg_instance = GenericFailureDiffPreferencePairGenerator()

            return HIRPreferencePairGenerator(
                problem_generator=pg_instance,
                solution_generator=sg_instance,
                judge=jg_instance,
                feedback_generator=fg_instance,
                virtue_preference_pair_generator=vppg_instances,
                failure_type_preference_pair_generator=ftppg_instance,
                **kwargs,
            )

        except AttributeError as e:
            logger.error(f"Error creating HIRPreferencePairGenerator: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating HIRPreferencePairGenerator: {e}")
            return None


    @staticmethod
    def hirpo_eval_from_config(
        solution: str,
        problem_generator: str,
        judge: str,
        model_kwargs: dict[str, Any],
        **kwargs
    ) -> HIREvaluator | None:
        """Create an evaluator from config."""
        try:
            pg_class = registry.get_class(problem_generator)
            if issubclass(pg_class, HIRAbstractGeneratorLLM):
                pg_instance = pg_class(**model_kwargs)
            elif issubclass(pg_class, HIRAbstractGenerator):
                pg_instance = pg_class()
            else:
                raise TypeError(f"Invalid problem generator class: {pg_class} of type {type(pg_class)}")

            sg_instance = GenericSolutionGenerator(
                solution_class=registry.get_class(solution),
                **model_kwargs,
            )

            jg_class = registry.get_class(judge)
            if issubclass(jg_class, HIRAbstractGeneratorLLM):
                jg_instance = jg_class(**model_kwargs)
            elif issubclass(jg_class, HIRAbstractGenerator):
                jg_instance = jg_class()
            else:
                raise TypeError(f"Invalid judge class: {jg_class} of type {type(jg_class)}")

            return HIREvaluator(
                problem_generator=pg_instance,
                solution_generator=sg_instance,
                judge=jg_instance,
                **kwargs,
            )

        except AttributeError as e:
            logger.error(f"Error creating evaluator: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating evaluator: {e}")
            return None
