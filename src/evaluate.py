

import pandas as pd

import pickle
from sklearn.metrics import accuracy_score

import yaml
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/yehdihouldanna/mlpipeline-mlflow-dvc.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = 'yehdihouldanna'
os.environ["MLFLOW_TRACKING_PASSWORD"]= "28fdecc4cae73c08dadfd562c7bfe688f20b9a78"

# load
train_params = yaml.safe_load(open("params.yaml"))['train']
dagshub_params = yaml.safe_load(open("params.yaml"))['dagshub']


def evaluate(data_path,model_path):
    data =pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(dagshub_params["MLFLOW_TRACKING_URI"])

    #? model = pickle.load(open(model_path,'rb'))
    import joblib

    model = joblib.load(model_path)

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("eval_accuracy",accuracy)

    print(f"Model acccuracy :{accuracy}")

if __name__=="__main__":
    evaluate(train_params["data"],train_params["model_path"])