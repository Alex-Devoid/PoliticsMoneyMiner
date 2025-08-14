import pytest
import os
import subprocess
from fastapi.testclient import TestClient
from cloud_function.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def firestore_emulator():
    """
    This fixture does nothing special; it simply exists to show we
    have a fixture for any future setup or teardown needs.

    We rely on get_firestore_client() to load environment variables
    from the .env file (or system env) using load_dotenv().
    """
    yield  # no changes to environment variables

@pytest.fixture(scope="module")
def real_id_token():
    """
    Use 'gcloud' to obtain a real identity token from the local environment,
    e.g., after running 'gcloud auth login'. This requires gcloud to be
    installed and authorized inside the container or dev environment.
    """
    try:
        token_bytes = subprocess.check_output(["gcloud", "auth", "print-identity-token"])
        real_token = token_bytes.strip().decode("utf-8")
        print("Obtained identity token from gcloud.")
        return real_token
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to obtain identity token via gcloud: {e}")
    except FileNotFoundError:
        pytest.fail("gcloud command not found. Make sure gcloud is installed in this environment.")

def test_send_endpoint_with_existing_doc(firestore_emulator, real_id_token, monkeypatch):
    """
    Calls POST /alerts/send using a real ID token from gcloud. This bypasses
    fetch_id_token(...) so we don't rely on metadata server or local ADC.

    Assumes you already have a doc in 'alerts_subscriptions' like:
      {
        "email": "alexander.d.devoid@gmail.com",
        "states": {
          "pa": {
            "places": {
              "collegetownship": {
                "committees": {
                  "College Township Council Meeting Agendas & Minutes": { "last_sent_at": null },
                  "Planning Commission": { "last_sent_at": null }
                }
              }
            }
          }
        }
      }

    Also assumes your environment (.env) is set so that:
      - FIRESTORE_EMULATOR_HOST=firestore-emulator:8180
      - GOOGLE_CLOUD_PROJECT=dev-emulated-project
      - GMAIL_MICROSERVICE_URL=http://127.0.0.1:9090/send-email
    """

    # Monkeypatch the function that fetches ID token so it just returns our real token
    monkeypatch.setattr("cloud_function.utils.email_service.get_identity_token",
                        lambda audience: real_id_token)

    # Now run the normal test
    response = client.post("/alerts/send")
    assert response.status_code == 200, f"Unexpected status: {response.status_code}, body: {response.text}"

    data = response.json()
    # Typically your endpoint returns: { "detail": "Hierarchical alerts sent successfully." }
    # or something similar. Adjust as needed.
    assert "detail" in data
    assert "successfully" in data["detail"].lower()
    # We confirm it didn't fail. 
    # If there's no new doc in 'scraped_meetings' => no actual alerts are sent,
    # but the endpoint still works and returns 200.
