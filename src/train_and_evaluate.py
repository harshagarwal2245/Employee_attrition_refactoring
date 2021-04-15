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
from get_data import read_params
import argparse
import joblib
import json


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
    n_estimator = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    max_feature = config["estimators"]["RandomForestClassifier"]["params"]["max_feature"]

    rf = RandomForestClassifier(
        n_estimators=n_estimator,
        max_features=max_feature,
        random_state=random_state)
    rf.fit(train_x, train_y)

    predicted_qualities = rf.predict(test_x)

    (acc, cm, rf) = eval_metrics(test_y, predicted_qualities)

    print("Random forest model (n_estimator=%f, max_feature=%f):" %
          (n_estimator, max_feature))
    print("  RMSE: %s" % acc)
    print("  MAE: %s" % cm)
    print("  R2: %s" % rf)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "acc": acc

        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimator": n_estimator,
            "max_feature": max_feature
        }

        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(rf, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
