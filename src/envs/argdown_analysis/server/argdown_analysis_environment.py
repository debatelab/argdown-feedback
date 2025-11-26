# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Argdown Analysis Environment Implementation.

A text environment for analysing arguments with Argdown.
"""

import logging
import os
import random
import time
from typing import Callable
from uuid import uuid4

import httpx  # For type checking
from openenv_core.env_server.interfaces import Environment  # type: ignore[import]
from openenv_core.env_server.types import Action, State  # type: ignore[import]

from argdown_feedback.api.shared.models import VerificationResponse
from argdown_feedback.api.client import (
    VerifiersClient,
    create_arganno_request,
    create_argmap_request,
    create_infreco_request,
    create_logreco_request,
    create_arganno_argmap_request,
    create_arganno_infreco_request,
    create_arganno_logreco_request,
    create_argmap_infreco_request,
    create_argmap_logreco_request,
    create_arganno_argmap_logreco_request,
)

from models import (
    ArgdownAnalysisStep,
    ArgdownAnalysisTask,
    ArgdownAnalysisAction,
    ArgdownAnalysisObservation,
    ArgdownAnalysisState,
)

from server.instructions import get_base_instruction

# Configure logging
logger = logging.getLogger(__name__)


MULTI_ARGUMENT_FLOW = {
    "arganno": ["arganno_argmap", "arganno_infreco", "arganno_logreco"],
    "argmap": ["arganno_argmap", "argmap_infreco", "argmap_logreco"],
    "arganno_argmap": ["arganno_argmap_logreco"],
    "arganno_infreco": ["arganno_logreco"],
    "arganno_logreco": ["arganno_argmap_logreco"],
    "argmap_infreco": ["argmap_logreco"],
    "argmap_logreco": ["arganno_argmap_logreco"],
}

SINGLE_ARGUMENT_FLOW = {
    "arganno": ["arganno_infreco", "arganno_logreco"],
    "infreco": ["arganno_infreco", "logreco"],
    "logreco": ["arganno_logreco"],
    "arganno_infreco": ["arganno_logreco"],
}


class ArgdownAnalysisEnvironment(Environment):
    """
    A text environment that guides and verifies logical argument analysis with Argdown.

    This environment is designed for verifying Argdown argument reconstructions.
    It maintains a source text to-be-analysed, breaks down the reconstruction into subtasks,
    and verifies solutions at each step.

    Example:
        >>> env = ArgdownAnalysisEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Argdown Analysis environment ready!"
        >>>
        >>> obs = env.step(ArgdownAnalysisAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    def __init__(
        self,
        argdown_feedback_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        backoff_factor: float = 2.0,
    ):
        """Initialize the argdown_analysis environment.
        
        Args:
            argdown_feedback_url: Base URL for argdown feedback API.
                If None, reads from ARGDOWN_FEEDBACK_URL environment variable.
            max_retries: Maximum number of retry attempts for API calls (default: 3)
            timeout: Request timeout in seconds (default: 30.0)
            backoff_factor: Exponential backoff multiplier for retries (default: 2.0)
        """
        # Get URL from parameter or environment variable
        url = argdown_feedback_url or os.getenv("ARGDOWN_FEEDBACK_URL")
        if not url:
            raise ValueError(
                "argdown_feedback_url must be provided either as parameter or "
                "via ARGDOWN_FEEDBACK_URL environment variable"
            )
                
        # Initialize client in sync mode for simpler synchronous environment
        self._verifiers_client = VerifiersClient(base_url=url, async_client=False, timeout=timeout)
        self._max_retries = max_retries
        self._timeout = timeout
        self._backoff_factor = backoff_factor
        
        # Health check - verify Argdown Feedback API is accessible
        try:
            # Type assertion: client is httpx.Client in sync mode
            client = self._verifiers_client.client
            assert isinstance(client, httpx.Client), "Client should be in sync mode"
            
            response = client.get(f"{url}/health")
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "healthy":
                raise RuntimeError(
                    f"API health check returned unexpected status: {data.get('status')}"
                )
            logger.info(
                f"Successfully connected to argdown-feedback API at {url} "
                f"(service: {data.get('service', 'unknown')}, "
                f"version: {data.get('version', 'unknown')})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to argdown-feedback API at {url}. "
                f"Ensure the API server is running and accessible."
            ) from e

        self._state = ArgdownAnalysisState(
            episode_id=str(uuid4()),
            step_count=0,
            source_text="",
            task_id=ArgdownAnalysisTask.SingleArgumentAnalysis,
            subtask_id="",
        )
        self._reset_count = 0
        
        logger.info(
            f"Initialized ArgdownAnalysisEnvironment with URL={url}, "
            f"max_retries={max_retries}, timeout={timeout}"
        )

    def reset(
        self, source_text: str | None = None, task_id: ArgdownAnalysisTask | None = None
    ) -> ArgdownAnalysisObservation:
        """
        Reset the environment.

        Returns:
            ArgdownAnalysisObservation with an initial argument reconstruction prompt
        """
        if source_text is None:
            raise ValueError("source_text must be provided for reset()")
        if task_id is None:
            task_id = ArgdownAnalysisTask.SingleArgumentAnalysis
        subtask_id=self._get_next_subtask(task_id=task_id)

        self._state = ArgdownAnalysisState(
            episode_id=str(uuid4()),
            step_count=0,
            subtask_step_count=0,
            source_text=source_text or "",
            task_id=task_id,
            subtask_id=subtask_id,
        )
        self._reset_count += 1

        prompt = self._get_next_instruction()
        self._state.history.append(ArgdownAnalysisStep(subtask_id=subtask_id, prompt=prompt, message="", verification_response=None))

        return ArgdownAnalysisObservation(
            prompt=prompt,
            done=False,
            reward=0.0,
        )

    def step(self, action: Action) -> ArgdownAnalysisObservation:
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: ArgdownAnalysisAction containing the message to echo

        Returns:
            ArgdownAnalysisObservation with the echoed message and its length
        """
        if not isinstance(action, ArgdownAnalysisAction):
            raise ValueError("Action must be an instance of ArgdownAnalysisAction.")

        self._state.step_count += 1
        self._state.subtask_step_count += 1

        message = action.message

        handler_fn: Callable[[str], tuple[float, VerificationResponse]]
        match self._state.subtask_id:
            case "arganno":
                handler_fn = self._handle_arganno
            case "argmap":
                handler_fn = self._handle_argmap
            case "infreco":
                handler_fn = self._handle_infreco
            case "logreco":
                handler_fn = self._handle_logreco
            case "arganno_argmap":
                handler_fn = self._handle_arganno_argmap
            case "arganno_infreco":
                handler_fn = self._handle_arganno_infreco
            case "arganno_logreco":
                handler_fn = self._handle_arganno_logreco
            case "argmap_infreco":
                handler_fn = self._handle_argmap_infreco
            case "argmap_logreco":
                handler_fn = self._handle_argmap_logreco
            case "arganno_argmap_logreco":
                handler_fn = self._handle_arganno_argmap_logreco
            case _:
                raise ValueError(f"Unknown subtask_id: {self._state.subtask_id}")

        reward, verification_response = handler_fn(message)

        # update current task / subtask state
        done = False
        if verification_response.is_valid:
            # subtask completed
            if self._state.subtask_id:
                self._state.subtasks_completed.append(self._state.subtask_id)
            next_subtask = self._get_next_subtask(
                task_id=self._state.task_id,
                subtasks_completed=self._state.subtasks_completed,
            )
            done = next_subtask is None
            self._state.subtask_id = next_subtask
            self._state.subtask_step_count = 0

        # generate next instruction prompt
        prompt = self._get_next_instruction(verification_response)

        # update history
        last_entry = self._state.history.pop()
        self._state.history.extend([
            ArgdownAnalysisStep(
                subtask_id=last_entry.subtask_id,
                prompt=last_entry.prompt,
                message=message,
                verification_response=verification_response.model_dump(),  # Serialize to dict
            ),
            ArgdownAnalysisStep(
                subtask_id=self._state.subtask_id,
                prompt=prompt,
                message="",
                verification_response=None,
            )
        ])

        return ArgdownAnalysisObservation(
            prompt=prompt,
            done=done,
            reward=reward,
            metadata={
                "verification_response": verification_response.model_dump(),  # Serialize to dict
                "original_message": message,
                "task": self._state.task_id,
                "subtask": self._state.subtask_id,
                "step": self._state.step_count,
                "subtask_step": self._state.subtask_step_count,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _get_next_instruction(
        self, last_verification_response: VerificationResponse | None = None
    ) -> str:
        """Generate the next instruction prompt based on the current subtask."""
        current_subtask = self._state.subtask_id
        if current_subtask is None:
            return "All subtasks completed. Well done!"

        feedback = ""
        if last_verification_response is not None:
            if last_verification_response.is_valid:
                feedback = "Your last submission was valid. Well done! We will now proceed with a more detailed analysis, referring back to the same source text as analysed before."
            else:
                error_messages = "\n\n".join([r.message for r in last_verification_response.results if r.message])
                feedback = "Your last submission was invalid. Please try again. Errors found:\n\n" + error_messages
                return feedback


        base_instruction = get_base_instruction(
            task_id=self._state.task_id,
            subtask_id = current_subtask,
            source_text = self._state.source_text,
        )

        if feedback:
            return f"{feedback}\n\n{base_instruction}"
        else:
            return base_instruction

    def _verify_with_retry(
        self,
        verifier_id: str,
        request,  # VerificationRequest object from argdown-feedback
        subtask_id: str,
    ) -> VerificationResponse:
        """Call verification API with retry logic and error handling.
        
        Args:
            verifier_id: ID of the verifier to use
            request: Verification request payload (pre-built VerificationRequest object)
            subtask_id: Current subtask identifier for error messages
            
        Returns:
            VerificationResponse from the API
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(
                    f"API call attempt {attempt + 1}/{self._max_retries + 1} "
                    f"for verifier={verifier_id}, subtask={subtask_id}"
                )
                
                # Make the API call
                response = self._verifiers_client.verify_sync(
                    verifier_id,
                    request,
                    # Note: If verify_sync doesn't support timeout, 
                    # this needs to be handled at the HTTP client level
                )
                
                logger.info(
                    f"API call successful for verifier={verifier_id}, "
                    f"is_valid={response.is_valid}, attempt={attempt + 1}"
                )
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self._max_retries + 1}): "
                    f"{type(e).__name__}: {str(e)}"
                )
                
                # Don't retry on last attempt
                if attempt < self._max_retries:
                    # Exponential backoff
                    wait_time = self._backoff_factor ** attempt
                    logger.debug(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        error_msg = (
            f"API call failed after {self._max_retries + 1} attempts. "
            f"Last error: {type(last_exception).__name__}: {str(last_exception)}"
        )
        logger.error(error_msg)
        
        # Always re-raise - we cannot create a proper fallback response
        # because VerificationResponse has required fields we don't have access to
        raise RuntimeError(error_msg) from last_exception

    def _handle_arganno(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_arganno_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("arganno", request, "arganno")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra reward for density
            density_score = next((s.score for s in verification_response.scores if s.scorer_id == "annotation_density_scorer"), 0.0)
            reward += density_score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_argmap(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_argmap_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("argmap", request, "argmap")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra reward for faithfulness and scope
            faithfulness_score = next((s.score for s in verification_response.scores if s.scorer_id == "argmap_faithfulness_scorer"), 0.0)
            scope_score = next((s.score for s in verification_response.scores if s.scorer_id == "argmap_scope_scorer"), 0.0)
            reward += scope_score * faithfulness_score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_infreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_infreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("infreco", request, "infreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra reward for default scores
            product_score = 1.0
            for s in [s.score for s in verification_response.scores]:
                product_score *= s
            reward += product_score            
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_logreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_logreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("logreco", request, "logreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra reward for default scores
            product_score = 5.0
            for s in [s.score for s in verification_response.scores]:
                product_score *= s
            reward += product_score            
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_arganno_argmap(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_arganno_argmap_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("arganno_argmap", request, "arganno_argmap")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_arganno_infreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_arganno_infreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("arganno_infreco", request, "arganno_infreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    def _handle_arganno_logreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_arganno_logreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("arganno_logreco", request, "arganno_logreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response
    
    def _handle_argmap_infreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_argmap_infreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("argmap_infreco", request, "argmap_infreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response
    
    def _handle_argmap_logreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_argmap_logreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("argmap_logreco", request, "argmap_logreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response
    
    def _handle_arganno_argmap_logreco(self, message: str) -> tuple[float, VerificationResponse]:
        request = create_arganno_argmap_logreco_request(message, self._state.source_text).build()
        verification_response = self._verify_with_retry("arganno_argmap_logreco", request, "arganno_argmap_logreco")
        reward = 0.0 
        # reward each individual check passed
        for result in verification_response.results:
            if result.is_valid:
                reward += .1 
        # main reward for overall success
        if verification_response.is_valid:
            reward += 5.0
            # extra rewards for default scorers
            for s in verification_response.scores:
                reward += s.score
            # extra reward for concise thinking
            reward += self._concise_thinking_reward(message)

        return reward, verification_response

    @staticmethod
    def _get_next_subtask(
        task_id: ArgdownAnalysisTask | None = None,
        subtasks_completed: list[str] | None = None,
    ) -> str | None:
        """Returns None if there are no more subtasks."""
        if task_id is None:
            return None
        candidates = []
        if task_id == ArgdownAnalysisTask.SingleArgumentAnalysis:
            if not subtasks_completed:
                candidates = ["arganno", "infreco", "logreco"]
            else:
                candidates = SINGLE_ARGUMENT_FLOW.get(subtasks_completed[-1], [])
        elif task_id == ArgdownAnalysisTask.MultiArgumentAnalysis:
            if not subtasks_completed:
                candidates = ["arganno", "argmap", "infreco"]
            else:
                candidates = MULTI_ARGUMENT_FLOW.get(subtasks_completed[-1], [])

        return random.choice(candidates) if candidates else None

    @staticmethod
    def _concise_thinking_reward(message: str) -> float:
        """Reward for concise thinking based on message length."""
        # extract thinking section
        message_lower = message.lower()      
        thinking_start = message_lower.find("<think>")
        if thinking_start == -1:
            return 0.0
        thinking_end = message_lower[thinking_start:].find("</think>")
        if thinking_end == -1:
            return 0.0
        reward = min(1.0, 0.9 ** ((thinking_end - thinking_start)/500))
        return reward