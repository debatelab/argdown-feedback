import time
from concurrent.futures import ThreadPoolExecutor

def mock_verification_handler():
    """Simulate a verification handler with a delay."""
    time.sleep(1)  # Simulate processing time
    return {"status": "success"}

def test_threading_verification(api_client):
    """Test threading behavior of the verification service."""
    def send_request():
        response = api_client.post("/api/v1/verify/arganno", json={"config": {}, "inputs": "test", "source": "test"})
        assert response.status_code == 200

    # Simulate 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request) for _ in range(10)]
        for future in futures:
            future.result()  # Ensure all requests complete successfully