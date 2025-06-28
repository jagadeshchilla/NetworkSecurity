import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle
from sklearn.model_selection import GridSearchCV
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    """
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    """
    Write a dictionary content to a file in YAML format.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    

def save_numpy_array_data(file_path:str,array:np.array)->None:
    """
    Save numpy array data to a file.
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
def save_object(file_path:str,obj:object)->None:
    """
    Save object data to a file.
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
def load_object(file_path:str)->object:
    """
    Load object data from a file.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
def load_numpy_array_data(file_path:str)->np.array:
    """
    Load numpy array data from a file.
    """
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    
   
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for model_name in models.keys():
            model=models[model_name]
            para=params[model_name]
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_model_score=get_classification_score(y_true=y_train,y_pred=y_train_pred)
            test_model_score=get_classification_score(y_true=y_test,y_pred=y_test_pred)
            report[model_name]=test_model_score
        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e
    