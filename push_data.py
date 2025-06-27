import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

## certifi is used to verify the ssl certificate and it is used to connect to the mongodb server
import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def cv_to_json(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_to_mongodb(self,records,database,collection_name):

        try:
            self.database=database
            self.collection_name=collection_name
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_client[self.database]
            self.collection_name=self.database[self.collection_name]
            self.collection_name.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
if __name__=="__main__":
    FILE_PATH="Network_data\phisingData.csv"
    DATABASE="jagadesh"
    COLLECTION_NAME="NetworkData"
    network_data_extract=NetworkDataExtract()
    records=network_data_extract.cv_to_json(file_path=FILE_PATH)
    no_of_records=network_data_extract.insert_data_to_mongodb(records=records,database=DATABASE,collection_name=COLLECTION_NAME)
    print(records)
    print(f"no of records inserted: {no_of_records}")






