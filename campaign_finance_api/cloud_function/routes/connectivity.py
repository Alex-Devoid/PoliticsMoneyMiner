from fastapi import APIRouter
import requests
import logging

router = APIRouter()

@router.get("/")
def check_connectivity():
    """
    Check connectivity to external services.
    """
    try:
        response_github = requests.get("https://api.github.com/")
        response_ip = requests.get("https://ifconfig.me")

        if response_github.status_code == 200 and response_ip.status_code == 200:
            return {"message": "Connectivity check successful."}
        else:
            return {
                "message": "Connectivity check failed.",
                "github_status": response_github.status_code,
                "ip_status": response_ip.status_code,
            }
    except requests.exceptions.RequestException as e:
        logging.error(f"Connectivity check failed: {e}")
        return {"error": str(e)}
