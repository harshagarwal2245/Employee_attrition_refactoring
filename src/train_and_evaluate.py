"""
load train and test file
train agorithm
save metric and paramters
"""
import os
#import warning
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
from get_data import read_params
import argparse
import joblib
import json
import mlflow


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    cm = confusion_matrix(actual, pred)
    cr = classification_report(actual, pred)
    return accuracy, cm, cr


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    target = config["base"]["target_col"]
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_y = train_data[target]
    test_y = test_data[target]

    train_x = train_data.drop(target, axis=1)
    test_x = test_data.drop(target, axis=1)
    ###################################Mlflow######################3
    # setting mlflow configuration

    mlflow_config=config["mlflow_config"]
    remote_server_uri=mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    # if we don't define wxpweimwnt name it will set default name
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
   
        n_estimator = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
        max_feature = config["estimators"]["RandomForestClassifier"]["params"]["max_feature"]

        rf = RandomForestClassifier(
            n_estimators=n_estimator,
            max_features=max_feature,
            random_state=random_state)
        rf.fit(train_x, train_y)

        predicted_qualities = rf.predict(test_x)

        (acc, cm, rf) = eval_metrics(test_y, predicted_qualities)

        ###############################################################3    
        """for logging values of metric and features we previously
        files i.e. json but mlfow provides us and another method i.e
        log_params and log metrics"""
        
        mlflow.log_param("n_estimator",n_estimator)
        mlflow.log_param("max_feature",max_feature)
        mlflow.log_metric("acc",acc)
        
        """
	We will check if server is on if it is not it will create and folder
	and then store information in it we will check scheme if scheme is file
	""" 
        tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(rf,"model",registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(rf,"model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
