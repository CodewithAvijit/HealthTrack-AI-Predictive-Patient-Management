import pandas as pd
from sklearn.feature_selection import f_classif,SelectKBest
import logging
import os
from utils import logger_add
import yaml
logger=logger_add("logs","feature_selection")
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

def feature_selection(data,n):
    try:
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1:]
        fc=SelectKBest(score_func=f_classif,k=n)
        df=fc.fit_transform(x,y)
        df=pd.DataFrame(df,columns=x.columns[fc.get_support()])
        df=pd.concat([df,y],axis=1)
        logger.debug(f"SELECTED FEATURES: [{df.columns.tolist()}]")
        return df
    except Exception as e:
        logger.error("ERROR ",e)
        raise
def save_data(traindata,testdata,path):
    try:
        dirname="featured"
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
        params=load_params("params.yaml")
        features=params['feature_selection']['features']
        trainpath="./process/interim/train.csv"
        testpath="./process/interim/test.csv"
        savepath="./process"
        traindata=load_data(trainpath)
        testdata=load_data(testpath)
        featuretrain=feature_selection(traindata,features)
        featuretest=testdata[featuretrain.columns.tolist()]
        save_data(featuretrain,featuretest,savepath)
        logger.debug("FEATURE SELECTION COMPLETED ")
    except  Exception as e:
        logger.error("ERROR ",e)
        raise


if __name__ == "__main__":
    main()
