# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Argdown Analysis Environment.

The argdown_analysis environment is an environment for analysing arguments with Argdown.
"""

from dataclasses import dataclass
from enum import Enum

from openenv_core.env_server.types import Action, Observation, State  # type: ignore[import]

from argdown_feedback.api.shared.models import VerificationResponse


class ArgdownAnalysisTask(Enum):
    SingleArgumentAnalysis = "SingleArgumentAnalysis"
    MultiArgumentAnalysis = "MultiArgumentAnalysis"


@dataclass(kw_only=True)
class ArgdownAnalysisAction(Action):
    """Action for the Argdown Analysis environment - just a message with Argdown snippets."""

    message: str
 

@dataclass(kw_only=True)
class ArgdownAnalysisObservation(Observation):
    """Observation from the Argdown Analysis environment - argdown feedback."""

    prompt: str


@dataclass(kw_only=True)
class ArgdownAnalysisState(State):
    """State of the Argdown Analysis environment."""

    source_text: str
    task_id: ArgdownAnalysisTask
    subtask_id: str | None
    subtasks_completed: list[str] = []
    subtask_step_count: int = 0
    history: list[tuple[str | None, str, str, VerificationResponse | None]] = [] 



