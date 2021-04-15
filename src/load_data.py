"""
read data from datasource and safe it to
data raw for further preprocess
"""
import os
from get_data import read_params, get_data
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df = df.drop(["StandardHours", "Over18", "EmployeeCount"], axis=1)
    categorical_variable = []
    numerical_variables = []
    for i in df.columns:
        if(df[i].dtypes == np.int64 or df[i].dtypes == np.float64):
            numerical_variables.append(i)
        else:
            categorical_variable.append(i)
    encoder = LabelEncoder()
    for i in categorical_variable:
        df[i] = encoder.fit_transform(df[i])
    df.to_csv(raw_data_path, index=False)


 # run comment
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
