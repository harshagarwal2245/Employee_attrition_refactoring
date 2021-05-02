"""
read data from datasource and safe it to
data raw for further preprocess
"""
import os
from get_data import read_params, get_data
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def feature_selections(x,y):
    test = SelectKBest(score_func=chi2, k=16)##we will get 16 columns automatically selected
    fit = test.fit(x, y)
    # summarize scores
    np.set_printoptions(precision=3)##upto 3 decimal accuracy not compulsary
    #print(fit.scores_)
    features = fit.transform(x)##coverts to matrix
    # summarize selected features
    #print(features)
    df1=pd.DataFrame(features)
    #print(df1.head())
    df1.columns=["Age","DailyRate","DistanceFromHome","EmployeeNumber","JobLevel","JobRole","MaritialStatus","MonthlyIncome","MonthlyRate","Overtime","StockOptionalLevel","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion"]
    df1["Attrition"]=y
    return df1



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
    x=df.drop("Attrition",axis=1)
    y=df["Attrition"]
    df=feature_selections(x,y)
    df.to_csv(raw_data_path, index=False)


 # run comment
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
