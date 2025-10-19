import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from utils import logger_add
import yaml

logger=logger_add("logs","data_ingestion")

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(path):
    try: 
        data=pd.read_csv(path)
        logger.debug("DATA LOAD SUCCESSFULLy")
        return data
    except Exception as e:
        logger.error("ERROR: ",e)
        raise
def split_data(data,test_size):
    try:
        traindata,testdata=train_test_split(data,test_size=test_size,random_state=42)
        logger.debug(f"SPLIT DATA INTO  TRAIN {(1-test_size)*100}%  AND TEST {test_size*100}%")
        return traindata,testdata
    except Exception as e:
        logger.error("ERROR ",e)
        raise

def save_data(traindata,testdata,path):
    try:
        dirname="raw"
        dirpath=os.path.join(path,dirname)
        os.makedirs(dirpath,exist_ok=True)
        traindata.to_csv(os.path.join(dirpath,"train.csv"),index=False)
        testdata.to_csv(os.path.join(dirpath,"test.csv"),index=False)
        logger.debug(f"RAW DATA SAVED IN {dirname}")
    except Exception as e:
        logger.error("ERROR ",e)
        raise

def main():
    try:
        params=load_params(r"params.yaml")
        testsize=params['data_ingestion']['test_size']
        path="./data/healthcare_dataset.csv"
        savepath="./process"
        data=load_data(path)
        traindata,testdata=split_data(data,testsize)
        save_data(traindata,testdata,savepath)
        logger.debug("DATA INGESTION COMPLETED ")
    except  Exception as e:
        logger.error("ERROR ",e)
        raise


if __name__ == "__main__":
    main()
