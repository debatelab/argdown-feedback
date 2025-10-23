#!/usr/bin/env python3
"""Simple test to see actual response data."""

import json
import requests
import subprocess
import time
import signal
import os
from contextlib import contextmanager

@contextmanager
def dev_server():
    """Start and stop the dev server."""
    # Start the development server
    proc = subprocess.Popen([
        "/Users/ggbetz/.local/bin/uv", "run", "python", "-m", 
        "argdown_feedback.api.server.dev_server"
    ], cwd="/Users/ggbetz/git/argdown-feedback")
    
    # Give it time to start
    time.sleep(3)
    
    try:
        yield "http://localhost:8000"
    finally:
        # Kill the server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def test_verifier():
    with dev_server() as base_url:
        # Test data
        invalid_support_ref_xml = """
        <proposition id="1">We should stop eating meat.</proposition>
        <proposition id="2" supports="3">Animals suffer.</proposition>
        """
        
        arganno_source_text = "We should stop eating meat. Animals suffer. Some animals are raised humanely."
        
        # Make API request
        response = requests.post(
            f"{base_url}/api/v1/verify/arganno",
            json={
                "inputs": invalid_support_ref_xml,
                "source": arganno_source_text,
                "config": {}
            }
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response data:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_verifier()