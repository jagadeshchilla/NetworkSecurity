import os
import sys
import pandas as pd
import numpy as np
"""
defining common constants for training pipeline
"""
TARGET_COLUMN="Result"
TARGET_COLUMN_NAME="Result"
PIPELINE_NAME="phishing_detection"
ARTIFACT_DIR="artifacts"
FILE_NAME="phishing_detection.csv"
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")
SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"

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

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str="data_validation"
DATA_VALIDATION_VALID_DIR: str="validated"
DATA_VALIDATION_INVALID_DIR: str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str="report.yaml"
PREPROCESSOR_OBJECT_FILE_NAME="preprocessor.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str="transformed_object"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str="train.csv"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str="test.csv"

## knn imputer to replaces nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict= {
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights":"uniform",
}

## preprocessor object file name
OBJECT_FILE_NAME: str="preprocessor.pkl"

"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str="trained_model"
MODEL_FILE_NAME: str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float=0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float=0.05

