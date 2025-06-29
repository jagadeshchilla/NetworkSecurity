import sys,os

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn
import dagshub
import os

# Initialize DagsHub and MLflow (only if not in production/Docker environment)
MLFLOW_ENABLED = os.getenv('MLFLOW_ENABLED', 'false').lower() == 'true'

if MLFLOW_ENABLED:
    try:
        dagshub.init(repo_owner='jagadeshchilla', repo_name='phishing-detecting', mlflow=True)
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri("https://dagshub.com/jagadeshchilla/phishing-detecting.mlflow")
        experiment_name = "phishing-detection-experiment"
        try:
            # Try to create experiment, if it exists, it will just get the existing one
            experiment = mlflow.set_experiment(experiment_name)
            logging.info(f"MLflow experiment '{experiment_name}' is ready")
        except Exception as e:
            logging.warning(f"Could not set experiment: {e}")
            MLFLOW_ENABLED = False
    except Exception as e:
        logging.warning(f"Could not initialize DagsHub/MLflow: {e}. Continuing without MLflow tracking.")
        MLFLOW_ENABLED = False
else:
    logging.info("MLflow tracking disabled (production mode)")


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def track_mlflow(self,model,classification_train_metric,classification_test_metric,model_name):
        if not MLFLOW_ENABLED:
            logging.info(f"MLflow tracking disabled, skipping logging for {model_name}")
            return
            
        try:
            with mlflow.start_run(run_name=f"phishing_detection_{model_name}"):
                # Log model name as a parameter
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_type", type(model).__name__)
                
                # Log training metrics
                train_f1_score=classification_train_metric.f1_score
                train_precision=classification_train_metric.precision_score
                train_recall=classification_train_metric.recall_score

                mlflow.log_metric("train_f1_score",train_f1_score)
                mlflow.log_metric("train_precision",train_precision)
                mlflow.log_metric("train_recall",train_recall)
                
                # Log test metrics
                test_f1_score=classification_test_metric.f1_score
                test_precision=classification_test_metric.precision_score
                test_recall=classification_test_metric.recall_score

                mlflow.log_metric("test_f1_score",test_f1_score)
                mlflow.log_metric("test_precision",test_precision)
                mlflow.log_metric("test_recall",test_recall)
                
                # Log additional metrics
                mlflow.log_metric("f1_score_diff", train_f1_score - test_f1_score)
                
                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=f"phishing_detection_{model_name}"
                )
                
                logging.info(f"Successfully logged {model_name} to MLflow")
                
        except Exception as e:
            logging.warning(f"Failed to log to MLflow: {e}. Continuing with model training.")

    def train_model(self,x_train,y_train,x_test,y_test):

        models={
            "RandomForest":RandomForestClassifier(verbose=1),
            "AdaBoost":AdaBoostClassifier(),
            "GradientBoosting":GradientBoostingClassifier(verbose=1),
            "LogisticRegression":LogisticRegression(verbose=1),
            "DecisionTree":DecisionTreeClassifier()
            
        }
        params={
            "DecisionTree":{
                'criterion':['gini','entropy','log_loss'],
            },
            "RandomForest":{
                'n_estimators':[8,16,32,64,128,256]
            },
            "GradientBoosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators':[8,16,32,64,128,256]
            },
            "LogisticRegression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators':[8,16,32,64,128,256]
            }
        }
        model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
        
        # Extract f1_scores for comparison
        model_scores = {model_name: metric.f1_score for model_name, metric in model_report.items()}
        best_moddel_score=max(model_scores.values())
        best_model_name = max(model_scores, key=model_scores.get)
        best_model=models[best_model_name]
        best_model.fit(x_train,y_train)
        y_train_pred=best_model.predict(x_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        
        ##Track the mlflow
        self.track_mlflow(best_model,classification_train_metric,classification_test_metric,best_model_name)
        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_model)

        save_object("final_model/model.pkl",best_model)
        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        
        return model_trainer_artifact

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    