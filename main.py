from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.components.model_trainer import ModelTrainer

import os
import sys



if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config=dataingestionconfig)
        logging.info("Inintiating data ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation completed successfully")
        print(data_ingestion_artifact)
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
        logging.info("Inintiating data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact=data_validation_artifact,
                                                data_transformation_config=data_transformation_config)
        logging.info("Inintiating data transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully")
        print(data_transformation_artifact)

        model_trainer_config=ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,
                                   data_transformation_artifact=data_transformation_artifact)
        logging.info("Inintiating model trainer")
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("Model trainer completed successfully")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)

