import os
import pytest
from fastapi.testclient import TestClient
from cloud_function.main import app
from google.cloud import firestore
import logging

# Set Firestore Emulator environment variables for testing
os.environ["FIRESTORE_EMULATOR_HOST"] = "firestore-emulator:8180"
os.environ["GOOGLE_CLOUD_PROJECT"] = "content-audit-333003"

# Test Client for FastAPI app
client = TestClient(app)

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s", force=True)

# Firestore collection name
COLLECTION_NAME = "scraped_documents"


@pytest.fixture
def firestore_client():
    """
    Fixture to provide a Firestore client connected to the emulator.
    """
    return firestore.Client()


def log_response(endpoint, params, response):
    """
    Helper function to log request and response details.
    """
    logging.debug(f"Request to endpoint: {endpoint} with params: {params}")
    logging.debug(f"Response status code: {response.status_code}")
    logging.debug(f"Response JSON: {response.json()}")


def test_get_meetings_filter_by_state():
    """
    Test filtering by state.
    """
    params = {"state": "pa"}
    response = client.get("/meetings", params=params)

    # Log the response
    log_response("/meetings", params, response)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] > 0  # Ensure there are results
    assert all(meeting["state"] == "pa" for meeting in data["data"])  # Verify all states are 'pa'


def test_get_meetings_filter_by_place():
    """
    Test filtering by place.
    """
    params = {"place": "durham"}
    response = client.get("/meetings", params=params)

    # Log the response
    # log_response("/meetings", params, response)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] > 0  # Ensure there are results
    assert all(meeting["place"] == "durham" for meeting in data["data"])  # Verify all places are 'durham'


def test_get_meetings_filter_by_date_range():
    """
    Test filtering by date range.
    """
    start_date = "2025-01-10"
    end_date = "2025-01-20"
    params = {"start_date": start_date, "end_date": end_date}
    response = client.get("/meetings", params=params)

    # Log the response
    # log_response("/meetings", params, response)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] > 0  # Ensure there are results
    for meeting in data["data"]:
        meeting_date = meeting["meeting_date"]
        assert start_date <= meeting_date <= end_date  # Verify date is within range


def test_get_meetings_filter_combined():
    """
    Test filtering by multiple parameters (state and place).
    """
    params = {"state": "nc"}
    response = client.get("/meetings", params=params)

    # Construct and log the full URL
    base_url = response.request.url
    print(f"Full URL: {base_url}")

    # Log the response
    log_response("/meetings", params, response)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["total"] > 0  # Ensure there are results
    for meeting in data["data"]:
        assert meeting["state"] == "nc"  # Verify state filter

