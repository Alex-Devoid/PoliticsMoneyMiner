import os
import requests
import logging
from typing import Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.id_token import fetch_id_token

def send_to_microservice(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends a JSON payload to the Cloud Run "gmail-microservice" with IAM authentication.
    
    In production, set GMAIL_MICROSERVICE_URL to your Cloud Run service root
      
    Then we append /send-email as the path.

    In local dev, if GMAIL_MICROSERVICE_URL is not set, we default to http://127.0.0.1:9090
    (assuming you're running 'gcloud run services proxy ... --port=9090').

    Returns the JSON response from the microservice.
    Raises an exception if the response is 4xx or 5xx.
    """


    # The base domain for your microservice (no trailing slash).
    # We'll append /send-email to this for the final request.
    microservice_base = os.getenv("GMAIL_MICROSERVICE_URL", "http://127.0.0.1:9090")
    endpoint_path = "/send-email"

    # Build the full URL for the POST
    final_url = f"{microservice_base.rstrip('/')}{endpoint_path}"

    logging.info(f"send_to_microservice: base={microservice_base}, path={endpoint_path}, final_url={final_url}")

    try:
        # The ID token audience must be the root domain (microservice_base), not including /send-email
        token = fetch_id_token(Request(), microservice_base)
    except Exception as e:
        logging.error(f"Failed to generate identity token for audience={microservice_base}: {e}")
        raise RuntimeError(f"Failed to generate identity token: {e}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logging.info(f"Sending payload to microservice: {payload}")
    resp = requests.post(final_url, json=payload, headers=headers)
    resp.raise_for_status()  # Raises an HTTPError if status >= 400
    return resp.json()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running local test to send an email via Gmail microservice...")

    # Sample email payload
    payload = {
        "recipients": ["alexander.d.devoid@gmail.com"],
        "subject": "Local Test from Python Script",
        "body": "<p>Hello from local test script!</p>",
        "csv_data": [],
        "send_individual": False,
        "use_batch": False
    }

    try:
        response = send_to_microservice(payload)
        logging.info("Email microservice responded with: %s", response)
    except Exception as exc:
        logging.error("Failed to send email: %s", exc)
        raise
