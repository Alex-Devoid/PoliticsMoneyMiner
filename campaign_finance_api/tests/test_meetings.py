import os
import pytest
import json
from fastapi.testclient import TestClient
from cloud_function.main import app
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter  # Import FieldFilter
# Set Firestore Emulator environment variables for testing


# Test Client for FastAPI app
client = TestClient(app)

# Firestore collection name
COLLECTION_NAME = "scraped_documents"
DATA_FILE = "tests/scraped_data.json"  # JSON file with the scraped data

@pytest.fixture
def firestore_client():
    """
    Fixture to provide a Firestore client connected to the emulator.
    """
    return firestore.Client()

def clear_collection(firestore_client):
    """
    Helper function to clear the scraped_documents collection.
    """
    docs = list(firestore_client.collection(COLLECTION_NAME).stream())
    for doc in docs:
        doc.reference.delete()

def test_get_meetings_endpoint(firestore_client):
    """
    Test the /meetings endpoint using data saved from the scrape test.
    """
    # 1. Clear leftover data
    clear_collection(firestore_client)

    # 2. Load data from JSON file
    assert os.path.exists(DATA_FILE), f"Scraped data file not found: {DATA_FILE}"
    with open(DATA_FILE, "r") as file:
        scraped_data = json.load(file)

    # 3. Insert scraped data into Firestore
    for item in scraped_data:
        firestore_client.collection(COLLECTION_NAME).document(item["meeting_id"]).set(item)

    # 4. Validate the /meetings endpoint
    response = client.get("/meetings/")
    if response.status_code != 200:
        print(f"Error Response: {response.json()}")
    assert response.status_code == 200
    response_data = response.json()
    print(response_data)
    assert response_data["total"] == len(scraped_data), "Mismatch in total meetings count"
    
    # # 5. Validate some specific filters
    # #    a) state="pa"
    # response = client.get("/meetings/", params={"state": "pa"})
    # assert response.status_code == 200
    # filtered_data = [item for item in scraped_data if item["state"] == "pa"]
    # assert response.json()["total"] == len(filtered_data)

    # #    b) Date range
    # response = client.get("/meetings/", params={"start_date": "2025-01-01", "end_date": "2025-01-15"})
    # assert response.status_code == 200
    # filtered_data = [
    #     item for item in scraped_data if "2025-01-01" <= item["meeting_date"] <= "2025-01-15"
    # ]
    # assert response.json()["total"] == len(filtered_data)


def test_pagination(firestore_client):
    """
    Test pagination on the /meetings endpoint with mocked data.
    """
    # 1. Clear leftover data
    clear_collection(firestore_client)

    # 2. Insert mock data into Firestore (e.g. 100 documents)
    scraped_data = [
        {"meeting_id": str(i), "state": "ca", "meeting_date": "2025-01-01"}
        for i in range(100)
    ]
    for item in scraped_data:
        firestore_client.collection(COLLECTION_NAME).document(item["meeting_id"]).set(item)

    # 3. Test first page
    response = client.get("/meetings/", params={"page": 1, "limit": 10})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 10, "First page should have 10 items"
    assert response.json()["total"] == 100, "Total should be 100"

    # 4. Test second page
    response = client.get("/meetings/", params={"page": 2, "limit": 10})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 10, "Second page should have 10 items"

    # 5. Test out-of-bounds page
    response = client.get("/meetings/", params={"page": 11, "limit": 10})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 0, "Out-of-bounds page should have 0 items"
