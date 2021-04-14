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
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_data import read_params
import argparse
import joblib 
import json   

def eval_metrics(actual, pred):
    rmse =0
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config=read_params(config_path)
    test_data_path=config["split_data"]["test_path"]
    train_data_path=config["split_data"]["train_path"]
    random_state= config["base"]["random_state"]
    model_dir= config["model_dir"]
    target = config["base"]["target_col"] 
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    train_y=train_data[target]
    test_y=test_data[target]

    train_x=train_data.drop(target,axis=1)
    test_x=test_data.drop(target,axis=1)
    n_estimator=config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    max_feature=config["estimators"]["RandomForestClassifier"]["params"]["max_feature"]

    rf = RandomForestClassifier(
        n_estimators=n_estimator, 
        max_features=max_feature, 
        random_state=random_state)
    rf.fit(train_x, train_y)

    predicted_qualities = rf.predict(test_x)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Random forest model (n_estimator=%f, max_feature=%f):" % (n_estimator, max_feature))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)







