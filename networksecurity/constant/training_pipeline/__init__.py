import os
import sys
import pandas as pd
import numpy as np
"""
defining common constants for training pipeline
"""
TARGET_COLUMN="Result"
PIPELINE_NAME="phishing_detection"
ARTIFACT_DIR="artifacts"
FILE_NAME="phishing_detection.csv"
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME

"""

DATA_INGESTION_COLLECTION_NAME: str="NetworkData"
DATA_INGESTION_DATABASE_NAME: str="jagadesh"
DATA_INGESTION_DIR_NAME: str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"
DATA_INGESTION_TRAIN_FILE_NAME: str="train.csv"
DATA_INGESTION_TEST_FILE_NAME: str="test.csv"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float=0.2






