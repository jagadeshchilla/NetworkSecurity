import sys
import os
import certifi
ca=certifi.where()
from dotenv import load_dotenv
load_dotenv()
mongo_db_url=os.getenv("MONGO_DB_URL")
print(mongo_db_url)
import pymongo


from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from fastapi import FastAPI,File,UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory="./templates")
database=client[DATA_INGESTION_DATABASE_NAME]
collection=database[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        training_pipeline=TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training completed successfully!!")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e


@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        preprocessor=load_object(file_path="final_model/preprocessor.pkl")
        model=load_object(file_path="final_model/model.pkl")
        network_model=NetworkModel(preprocessor=preprocessor,model=model)
        
        # Create a DataFrame copy without target column for prediction
        # Assuming the last column might be target, drop it if it exists
        prediction_df = df.copy()
        if 'Result' in prediction_df.columns:
            prediction_df = prediction_df.drop('Result', axis=1)
        
        print(f"Prediction DataFrame shape: {prediction_df.shape}")
        print(f"First row: {prediction_df.iloc[0]}")
        
        # Predict on the entire DataFrame (2D format)
        y_pred = network_model.predict(prediction_df)
        print(f"Predictions: {y_pred}")
        
        # Add predictions to the original DataFrame
        df["predictions"] = y_pred
        print(f"DataFrame with predictions shape: {df.shape}")
        
        # Save predictions
        df.to_csv("prediction_output/prediction.csv", index=False, header=True)
        
        # Create HTML table
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e



if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)