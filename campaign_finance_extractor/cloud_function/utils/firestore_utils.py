import os
import logging
from google.cloud import firestore

from typing import List, Dict

# Configure logging at the DEBUG level
logging.basicConfig(level=logging.DEBUG)

def get_firestore_client():
    """Initializes and returns a Firestore client."""
    emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST","")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "dev-emulated-project")
    database_id = os.getenv("FIRESTORE_DATABASE_ID", "(default)")

    logging.debug(f"Environment Variables: FIRESTORE_EMULATOR_HOST={emulator_host}, "
                  f"GOOGLE_CLOUD_PROJECT={project_id}, FIRESTORE_DATABASE_ID={database_id}")
    
    print(f"Environment Variables: FIRESTORE_EMULATOR_HOST={emulator_host}, "
                  f"GOOGLE_CLOUD_PROJECT={project_id}, FIRESTORE_DATABASE_ID={database_id}")    

    if emulator_host:
        print(f"Connecting to Firestore emulator at {emulator_host} (Project: {project_id})")
        # Use project_id from environment instead of hard-coded value
        client = firestore.Client(project=project_id)
    else:
        logging.info(f"Connecting to Firestore (Project: {project_id}, Database: {database_id})")
        client = firestore.Client(project=project_id, database=database_id)

    print(f"Firestore client initialized with project: {client.project}")
    return client

def write_documents_to_firestore(collection_name: str, documents: List[Dict]) -> None:
    """Writes a list of documents to a Firestore collection."""
    client = get_firestore_client()
    collection_ref = client.collection(collection_name)
    logging.info(f"Attempting to write {len(documents)} document(s) to collection '{collection_name}'.")

    for doc in documents:
        doc_id = doc.get('meeting_id')
        if doc_id:
            try:
                doc_ref = collection_ref.document(str(doc_id))
                doc_ref.set(doc)
                logging.info(f"✅ Document with ID {doc_id} written to collection '{collection_name}'.")
            except Exception as e:
                logging.error(f"❌ Failed to write document with ID {doc_id}: {e}")
        else:
            logging.warning("⚠️ Document skipped due to missing 'meeting_id'.")
