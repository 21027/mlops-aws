import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle 
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import boto3

# dagshub experiment tracking uri : found in experiment onglet
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/21027/Projet_ml.mlflow"

os.environ["MLFLOW_TRACKING_USERNAME"] = '21027' #dagsub username

# this is the token in your profile (default) and you can also copy it from the dvc onglet in the repository
os.environ["MLFLOW_TRACKING_PASSWORD"]= "958f3dd96ec3d6649dfc4d941243c3bc9a7ce45f"


def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=2,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])
    with mlflow.start_run():

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

        signature = infer_signature(X_train,y_train)

        # Define hyper params grid: 
        param_grid = {
            # 'n_estimators' : [100, 200],
            # 'max_depth' : [5,10,None],
            # 'min_samples_split' : [2,5],
            'min_samples_leaf' : [1,2]
        }

        # Logging experiment details : 
        mlflow.set_tag("mlflow.runName","Best RandomForestClassifier")
        mlflow.set_tag("experiment_type","grid_search_cv for best hyperparameters")
        mlflow.set_tag("model_type","RandomForestClassifier")
        mlflow.set_tag("description","RandomForestClassifier with GridSearchCV For Hyperparameter Tuning")

        # Perform hte hyperparams tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        # get the best model : 
        best_model = grid_search.best_estimator_

        # predict and evaluate the model

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_params(grid_search.best_params_)

        # log the confusions matrix and  classification report : 
        cm = confusion_matrix(y_test,y_pred)
        classification_rep = classification_report(y_test,y_pred,output_dict=True)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(str(classification_rep),"classification_report.txt")

        
        for label, metrics in classification_rep.items():
            if isinstance(metrics,dict): # for precision recall, f1-score, etc,
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}",value)

        conf_matrix = confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix,annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion_matrix")

        # artifacts
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # logging the model :
        mlflow.sklearn.log_model(best_model, "Best_RandomForestClassifier",
                                 registered_model_name="Best_RandomForestClassifier",
                                 signature=signature)


if __name__=="__main__":

    train_params = yaml.safe_load(open("params.yaml"))['train']
    mlflow_params = yaml.safe_load(open("params.yaml"))['mlflow']
    aws_params = yaml.safe_load(open("params.yaml"))['aws']
    # Set AWS credentials as environment variables for MLflow and boto3 to use
    os.environ['AWS_ACCESS_KEY_ID'] = aws_params['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_params['aws_secret_access_key']
    os.environ['AWS_DEFAULT_REGION'] = aws_params['region_name']

    train(train_params["data"],train_params["model_path"],train_params["random_state"], train_params["n_estimators"],train_params['max_depth'])