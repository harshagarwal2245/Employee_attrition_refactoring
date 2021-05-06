import json 
import logging 
import os
import joblib 
import pytest
from prediction_service.prediction import form_response,api_response 
import prediction_service 



input_data={
    "incorrect_range":{
        "Age":65,
        "DailyRate":200,
        "DistanceFromHome":20,
        "EmployeeNumber":200,
        "JobLevel":4,
        "JobRole":5,
        "MaritialStatus":4,
        "MonthlyIncome":2000,
        "MonthlyRate":5000,
        "Overtime":1,
        "StockOptionalLevel":2,
        "TotalWorkingYears":20,
        "TrainingTimesLastYear":20,
        "YearsAtCompany":10,
        "YearsInCurrentRole":10,
        "YearsSinceLastPromotion":5
    },
    "correct_range":{
        "Age":40,
        "DailyRate":200,
        "DistanceFromHome":20,
        "EmployeeNumber":200,
        "JobLevel":4,
        "JobRole":5,
        "MaritialStatus":2,
        "MonthlyIncome":2000,
        "MonthlyRate":5000,
        "Overtime":1,
        "StockOptionalLevel":2,
        "TotalWorkingYears":20,
        "TrainingTimesLastYear":20,
        "YearsAtCompany":10,
        "YearsInCurrentRole":10,
        "YearsSinceLastPromotion":5
    },
    "incorrect_col":{
        "Age":40,
        "DailyRate":200,
        "DistanceFromHome":20,
        "EmployeeNumber":200,
        "JobLevel":4,
        "JobRole":5,
        "MaritialStatus":2,
        "MonthlyIncome":2000,
        "MonthlyRate":5000,
        "Overtime":1,
        "StockOptionalLevel":2,
        "TotalWorkingYears":20,
        "TrainingTimeLastYear":20,
        "YearsAtCompany":10,
        "YearsInCurrentRole":10,
        "YearsSinceLastPromotion":5
    }
}
target_range={
"min":0,
"max":1
}

#def test_form_response_correct_range(data=input_data["correct_range"]):
#    res=int(form_response(data))
#    assert target_range["min"] <= res <= target_range["max"]


#def test_api_response_correct_range(data=input_data["correct_range"]):
#    res = api_response(data)
#    assert  target_range["min"] <= res["response"] <= target_range["max"]

#def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
#    with pytest.raises(prediction_service.prediction.NotInRange):
#       res = form_response(data) 


def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message