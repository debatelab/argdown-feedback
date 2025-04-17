import pytest

from argdown_feedback.tasks.base import (
    HIRPreferencePairGenerator,
    HIREvaluator,
)
from argdown_feedback.tasks.factory import HIRPOFactory
from .util import llm_available, MODEL_KWARGS


"""
solution: "arganno.Annotation"
    problem: "arganno.AnnotationProblem"
    problem_generator: "arganno.AnnotationProblemGenerator"
    judge: "arganno.AnnotationJudge"
    feedback_generator: "arganno.AnnotationFeedbackGenerator"
    virtue_preference_pair_generator: 
      - "arganno.AnnotationAttacksPreferencePairGenerator"
      - "arganno.AnnotationCoveragePreferencePairGenerator"
      - "arganno.AnnotationNoAttacksPreferencePairGenerator"
      - "arganno.AnnotationScopePreferencePairGenerator"
      - "arganno.AnnotationSupportsPreferencePairGenerator"
"""

@pytest.fixture
def config_data():
    return [
        {
            "solution": "arganno.Annotation",
            "problem": "arganno.AnnotationProblem",
            "problem_generator": "arganno.AnnotationProblemGenerator",
            "judge": "arganno.AnnotationJudge",
            "feedback_generator": "arganno.AnnotationFeedbackGenerator",
            "virtue_preference_pair_generator": [
                "arganno.AnnotationAttacksPreferencePairGenerator",
                "arganno.AnnotationCoveragePreferencePairGenerator",
                "arganno.AnnotationNoAttacksPreferencePairGenerator",
                "arganno.AnnotationScopePreferencePairGenerator",
                "arganno.AnnotationSupportsPreferencePairGenerator"
            ]
        }
    ]

@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
def test_hirgen_fromconfig(config_data):
    # Create an instance of HIRPreferencePairGenerator with the provided config data
    for config in config_data:
        hirgen = HIRPOFactory.hirpo_gen_from_config(
            **config,
            model_kwargs=MODEL_KWARGS,
        )
        # Check if the instance is created successfully
        assert isinstance(hirgen, HIRPreferencePairGenerator)

@pytest.mark.skipif(not llm_available(), reason="LLM model not available")
def test_hirevaluator_fromconfig(config_data):
    # Create an instance of HIREvaluator with the provided config data
    for config in config_data:
        hirgen = HIRPOFactory.hirpo_eval_from_config(
            **config,
            model_kwargs=MODEL_KWARGS,
        )
        # Check if the instance is created successfully
        assert isinstance(hirgen, HIREvaluator)