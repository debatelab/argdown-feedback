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

from typing import Any, Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from .models import ArgdownAnalysisAction, ArgdownAnalysisObservation


class ArgdownAnalysisEnv(HTTPEnvClient[ArgdownAnalysisAction, ArgdownAnalysisObservation]):
    """
    HTTP client for the Argdown Analysis Environment.

    This client connects to a ArgdownAnalysisEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = ArgdownAnalysisEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.echoed_message)
        >>>
        >>> # Send a message
        >>> result = client.step(ArgdownAnalysisAction(message="Hello!"))
        >>> print(result.observation.echoed_message)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ArgdownAnalysisEnv.from_docker_image("argdown_analysis-env:latest")
        >>> result = client.reset()
        >>> result = client.step(ArgdownAnalysisAction(message="Test"))
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
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
