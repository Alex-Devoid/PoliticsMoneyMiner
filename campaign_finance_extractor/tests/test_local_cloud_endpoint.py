import os
import sys
import time
import pytest
import asyncio
import uvicorn
import threading
from fastapi.testclient import TestClient
import requests

# Add the project root to sys.path so that Python can find the cloud_function module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cloud_function.main_sunset import app  # Import the FastAPI app

client = TestClient(app)

# Helper function to run FastAPI in a separate thread, simulating a Cloud Run container
def run_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=8081)

def test_process_newsletter_endpoint():
    """
    Integration test to verify the full workflow of the /process-newsletter endpoint.
    """
    try:
        # Start FastAPI server in a separate thread, simulating Cloud Run
        server_thread = threading.Thread(target=run_fastapi_server)
        server_thread.daemon = True
        server_thread.start()

        # Wait a bit for the server to start
        time.sleep(2)

        # Simulating Cloud Scheduler calling the /process-newsletter endpoint
        cloud_scheduler_url = "http://0.0.0.0:8081/process-newsletter"
        start_time = time.time()

        # Send a POST request to the /process-newsletter endpoint (like Cloxud Scheduler would)
        response = requests.post(cloud_scheduler_url, json={"force_reprocess": True})

        # End timer for performance analysis
        end_time = time.time()
        duration = end_time - start_time

        # Output the results of the test
        print(f"Integration Test Status Code: {response.status_code}")
        # print(f"Integration Test Response JSON: {response.json()}")
        print(f"Integration Test Duration: {duration} seconds")

        # Check if the workflow was successful
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        assert "Cases processed successfully" in response.json().get("message", ""), "Expected success message not found."

    except Exception as e:
        # If any exception occurs during the test, it will be caught and printed here.
        print(f"Integration test failed: {e}")

    finally:
        # Cleanup: Optionally, stop the server or kill the thread if necessary
        pass

if __name__ == "__main__":
    # Run the integration test
    test_process_newsletter_endpoint()
