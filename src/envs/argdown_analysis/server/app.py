# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Argdown Analysis Environment.

This module creates an HTTP server that exposes the ArgdownAnalysisEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app

Environment variables:
    ARGDOWN_CONFIG_PATH: Path to YAML configuration file (default: configs/default.yaml)
    MAX_RETRIES: Maximum retry attempts for verifier calls (default: 3)
    TIMEOUT: Request timeout in seconds (default: 30.0)
    BACKOFF_FACTOR: Exponential backoff multiplier (default: 2.0)
"""

import os
from pathlib import Path

try:
    from openenv_core.env_server.http_server import create_app  # type: ignore[import]
except Exception as e:  # pragma: no cover
    raise ImportError("openenv_core is required for the web interface. Install dependencies with '\n    uv sync\n'") from e

from .argdown_analysis_environment import ArgdownAnalysisEnvironment
from ..models import ArgdownAnalysisAction, ArgdownAnalysisObservation

# Get configuration from environment variables
config_path = os.getenv("ARGDOWN_CONFIG_PATH")  # Can be None, will use default
max_retries = int(os.getenv("MAX_RETRIES", "3"))
timeout = float(os.getenv("TIMEOUT", "30.0"))
backoff_factor = float(os.getenv("BACKOFF_FACTOR", "2.0"))

# Log configuration source
if config_path:
    print(f"Loading configuration from: {config_path}")
else:
    default_config = Path(__file__).parent.parent / "configs" / "default.yaml"
    print(f"Using default configuration: {default_config}")

# Create the environment instance
env = ArgdownAnalysisEnvironment(
    config_path=config_path,
    max_retries=max_retries,
    timeout=timeout,
    backoff_factor=backoff_factor,
)

# Create the app with web interface and README integration
app = create_app(
    env,
    ArgdownAnalysisAction,
    ArgdownAnalysisObservation,
    env_name="argdown_analysis",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m argdown_analysis.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn argdown_analysis.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
