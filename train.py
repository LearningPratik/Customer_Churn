import os
import re
import time
import yaml

from pathlib import Path
import mlflow
import pandas as pd
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.data.process import split_data, Ohe
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")

if __name__ == "__main__":
    # Set random seeds for reproducibility purpose
    seed = 42

    params_file = 'params.yaml'
    with open(params_file) as f:
        params = yaml.safe_load(f)


    # Create an experiment. By default, if not specified, the "default" experiment is used. It is recommended to not use
    # the default experiment and explicitly set up your own for better readability and tracking experience.
    client = MlflowClient()
    experiment_name = "Churn Classification"
    model_architecture = params['random_forest']['name']
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"churn_classification_rf_{timestamp}"

    run_name = model_name
    try:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    except MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Fetch experiment metadata information
    logger.info(f"Name: {experiment.name}")
    logger.info(f"Experiment_id: {experiment.experiment_id}")
    logger.info(f"Artifact Location: {experiment.artifact_location}")
    logger.info(f"Tags: {experiment.tags}")
    logger.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    
    # training_output_dir = os.path.join("./experiments/training_outputs", model_name)
    # checkpoints_dir = os.path.join(training_output_dir, "checkpoints")
    
    data = params['data_path']['path']
    X_train, X_test, y_train, y_test = split_data(data)
    # print(X_train.sample(), y_train.sample())
    print('Data split done..')

    ohe_preprocessor = Ohe()
    
    print('processing started')
    ohe_preprocessor.fit_transform(X_train)
    # print(ohe_preprocessor)
    print('processing done')
    
    max_depth    = params['random_forest']['max_depth']
    n_estimators = params['random_forest']['n_estimators']
    random_state = params['random_forest']['random_state']

    parameters = {'max_depth' : max_depth, 'n_estimators' : n_estimators, 'random_state' : random_state}

    rf_model = Pipeline(steps=[
        ('preprocessor', ohe_preprocessor),
        ('clf',RandomForestClassifier(**parameters))
    ])
    
    print('fitting data to model')
    rf_model.fit(X_train, y_train)
    print('model trained')

    # Make predictions
    print('making prediction..')
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred).item()
    recall = recall_score(y_test, y_pred).item()
    f1 = f1_score(y_test, y_pred).item()
    print('prediction and metrics calculated..')

    # y_pred = model_pipe.predict(X_test)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # plot confusion matrix
    
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print('created Confusion matrix')

    mlflow.sklearn.autolog()

        # Launch training phase
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        logger.info("tracking uri:", mlflow.get_tracking_uri())
        logger.info("artifact uri:", mlflow.get_artifact_uri())
        logger.info("start training")


        # log training parameters
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        
        # log the model
        mlflow.sklearn.log_model(sk_model = rf_model, input_example = X_test, artifact_path = "models")
        mlflow.log_figure(cm.figure_, artifact_file = 'confusion_matrix.png')

        # save dataset's dvc file
        mlflow.log_artifact(params['data_path']['path'])

        # rf_model.fit(X_train,y_train)
        # mlflow.log_artifacts(training_output_dir)
    
    print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))