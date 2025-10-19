from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_selection import f_classif,SelectKBest
import logging
import os
import joblib
from utils import logger_add

logger=logger_add("logs","model_training")

def load_data(path):
    try: 
        data=pd.read_csv(path)
        logger.debug("DATA LOAD SUCCESSFULLy")
        return data
    except Exception as e:
        logger.error("ERROR: ",e)
        raise

def model_training(data):
    try:
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1:]
        xg=RandomForestClassifier()
        xg.fit(x,y)
        logger.debug(f"MODEL TRAINING COMPLETED")
        return xg
    except Exception as e:
        logger.error("ERROR ",e)
        raise
def save_model(model,path):
    try:
        joblib.dump(model,os.path.join(path,"model.pkl"))
        logger.debug(f"MODEL SAVED IN {path}")
    except Exception as e:
        logger.error("ERROR ",e)
        raise
def main():
    try:
        trainpath="./process/featured/train.csv"
        savepath="./models"
        traindata=load_data(trainpath)
        model=model_training(traindata)
        save_model(model,savepath)
        logger.debug("Model Training stage COMPLETED ")
    except  Exception as e:
        logger.error("ERROR ",e)
        raise


if __name__ == "__main__":
    main()
