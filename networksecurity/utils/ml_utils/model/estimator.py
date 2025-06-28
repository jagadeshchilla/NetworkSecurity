from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import pickle

class NetworkModel:
    def __init__(self,preprocessor:object,model:object):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def predict(self,X):
        try:
            X=self.preprocessor.transform(X)
            y_pred=self.model.predict(X)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    @classmethod
    def load_object(cls,file_path:str)->object:
        try:
            if not os.path.exists(file_path):
                raise Exception(f"The file: {file_path} is not exists")
            with open(file_path,"rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
