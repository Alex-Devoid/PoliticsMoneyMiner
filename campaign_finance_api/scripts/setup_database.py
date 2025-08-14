#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import argparse
from cloud_function.database import Base, engine
from cloud_function.models_database import *  # Import all models to register them with Base

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_database(drop_existing: bool = False):
    """
    Sets up the database schema.

    Args:
        drop_existing (bool): Whether to drop existing tables before creating new ones.
    """
    try:
        if drop_existing:
            logging.info("Dropping existing tables...")
            Base.metadata.drop_all(bind=engine)
        logging.info("Creating new tables...")
        Base.metadata.create_all(bind=engine)
        logging.info("Database setup completed successfully!")
    except Exception as e:
        logging.error(f"Error setting up the database: {e}")
        sys.exit(1)  # Exit with an error status code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the database schema.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating new ones.",
    )
    args = parser.parse_args()

    logging.info("Starting database setup...")
    setup_database(drop_existing=args.drop_existing)
