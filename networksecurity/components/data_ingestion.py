from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact

## configuration of the data ingestion

from networksecurity.entity.config_entity import DataIngestionConfig
import os
import sys
import numpy as np
import pymongo
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def export_data_into_data_frame(self):
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def export_data_into_feature_store(self,data_frame:pd.DataFrame):

        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            data_frame.to_csv(feature_store_file_path,index=False,header=True)
            return feature_store_file_path
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def split_data_as_train_test(self,data_frame:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(
                data_frame,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            logging.info(f"performed train test split on the dataframe")
            logging.info("Exited split_as_train_test method of Data_Ingestion class")
            dir_path=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train set to {self.data_ingestion_config.train_file_path} and test set to {self.data_ingestion_config.test_file_path}")

            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
            logging.info(f"Exported train set and test set to {self.data_ingestion_config.train_file_path} and {self.data_ingestion_config.test_file_path} respectively")

            
            return train_set,test_set
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def intiate_data_ingestion(self):
        try:
            dataframe=self.export_data_into_data_frame()
            self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact=DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        