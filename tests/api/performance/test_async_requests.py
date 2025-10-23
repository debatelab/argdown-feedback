import asyncio
import time
import pytest

@pytest.mark.asyncio
async def test_performance_verification(api_client):
    """Test performance of the verification service under load."""
    async def send_request():
        start_time = time.time()
        response = await asyncio.to_thread(
            api_client.post, "/api/v1/verify/arganno", json={"config": {}, "inputs": "test", "source": "test"}
        )
        assert response.status_code == 200
        return time.time() - start_time

    # Simulate 50 concurrent requests
    start = time.time()
    tasks = [send_request() for _ in range(50)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start

    # Calculate average response time
    avg_response_time = sum(results) / len(results)
    print(f"Total time: {total_time:.2f}s, Average response time: {avg_response_time:.2f}s")