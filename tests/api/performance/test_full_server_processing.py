import time
import requests  # type: ignore
import pytest
import subprocess
import httpx  # type: ignore
import asyncio

# Helper function to start the server using uv

PATH_TO_UV = "/Users/ggbetz/.local/bin/uv"  # Adjust this path as needed

def start_server():
    with open("test_server.log", "w") as log_file:
        process = subprocess.Popen(
            [
                PATH_TO_UV, "run", "uvicorn", "--host", "127.0.0.1", "--port", "8001", "argdown_feedback.api.server.app:app"
            ],
            stdout=log_file,
            stderr=log_file
        )
        return process

@pytest.fixture(scope="module")
def server():
    # Start the server in a separate thread
    process = start_server()
    time.sleep(3)  # Wait for the server to start
    yield
    process.terminate()  # Ensure the server is terminated after the test

def test_sync_processing(server):
    """Test synchronous processing of the FastAPI server."""
    url = "http://127.0.0.1:8001/api/v1/verify/arganno"
    payload = {"config": {}, "inputs": "test", "source": "test"}

    start_time = time.time()
    for _ in range(10):  # Simulate 10 requests
        response = requests.post(url, json=payload)
        assert response.status_code == 200
    total_time = time.time() - start_time

    print(f"Total time for 10 requests: {total_time:.2f}s")

@pytest.mark.asyncio
async def test_async_processing(server):
    """Test asynchronous processing of the FastAPI server."""
    url = "http://127.0.0.1:8001/api/v1/verify/arganno"
    payload = {"config": {}, "inputs": "test", "source": "test"}

    async with httpx.AsyncClient() as client:
        start_time = time.time()
        tasks = [client.post(url, json=payload) for _ in range(10)]  # Simulate 10 concurrent requests
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        for response in responses:
            assert response.status_code == 200

    print(f"Total time for 10 async requests: {total_time:.2f}s")
