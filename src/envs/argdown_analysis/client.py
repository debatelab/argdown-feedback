# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Argdown Analysis Environment HTTP Client.

This module provides the client for connecting to a Argdown Analysis Environment server
over HTTP.
"""

from typing import Dict

from openenv_core.client_types import StepResult  # type: ignore[import]
from openenv_core.http_env_client import HTTPEnvClient  # type: ignore[import]

from .models import (
    ArgdownAnalysisAction,
    ArgdownAnalysisObservation,
    ArgdownAnalysisState,
    ArgdownAnalysisStep,
    ArgdownAnalysisTask,
)


class ArgdownAnalysisEnv(HTTPEnvClient[ArgdownAnalysisAction, ArgdownAnalysisObservation]):
    """
    HTTP client for the Argdown Analysis Environment.

    This client connects to an ArgdownAnalysisEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    The environment guides users through multi-step argument analysis tasks using
    Argdown notation, providing verification feedback at each step.

    Example:
        >>> # Connect to a running server
        >>> client = ArgdownAnalysisEnv(base_url="http://localhost:8000")
        >>> result = client.reset(source_text="Democracy is good.", task_id="SingleArgumentAnalysis")
        >>> print(result.observation.prompt)
        >>>
        >>> # Submit an argument reconstruction
        >>> action = ArgdownAnalysisAction(
        ...     message="<think>Analyzing...</think>\n```argdown\n<Arg>: Democracy\n```"
        ... )
        >>> result = client.step(action)
        >>> print(result.observation.prompt)
        >>> print(result.reward)
        >>> print(result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ArgdownAnalysisEnv.from_docker_image("argdown_analysis-env:latest")
        >>> result = client.reset(source_text="...", task_id="MultiArgumentAnalysis")
        >>> result = client.step(ArgdownAnalysisAction(message="..."))
    """

    def _step_payload(self, action: ArgdownAnalysisAction) -> Dict:
        """
        Convert ArgdownAnalysisAction to JSON payload for step request.

        Args:
            action: ArgdownAnalysisAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ArgdownAnalysisObservation]:
        """
        Parse server response into StepResult[ArgdownAnalysisObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with ArgdownAnalysisObservation
        """
        obs_data = payload.get("observation", {})
        observation = ArgdownAnalysisObservation(
            prompt=obs_data.get("prompt", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ArgdownAnalysisState:
        """
        Parse server response into ArgdownAnalysisState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            ArgdownAnalysisState object with full environment state
        """
        # Parse history - convert dict representations back to ArgdownAnalysisStep objects
        history_data = payload.get("history", [])
        history = [
            ArgdownAnalysisStep(
                subtask_id=step.get("subtask_id"),
                prompt=step.get("prompt", ""),
                message=step.get("message", ""),
                verification_response=step.get("verification_response"),  # Keep as dict
            )
            for step in history_data
        ]
        
        return ArgdownAnalysisState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            source_text=payload.get("source_text", ""),
            task_id=ArgdownAnalysisTask(payload.get("task_id", "SingleArgumentAnalysis")),
            subtask_id=payload.get("subtask_id"),
            subtasks_completed=payload.get("subtasks_completed", []),
            subtask_step_count=payload.get("subtask_step_count", 0),
            history=history,
        )
