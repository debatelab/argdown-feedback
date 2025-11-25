# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Argdown Analysis Environment.

The argdown_analysis environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class ArgdownAnalysisAction(Action):
    """Action for the Argdown Analysis environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class ArgdownAnalysisObservation(Observation):
    """Observation from the Argdown Analysis environment - the echoed message."""

    echoed_message: str
    message_length: int = 0

