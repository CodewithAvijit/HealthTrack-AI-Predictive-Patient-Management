import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import os
from utils import logger_add

logger=logger_add("logs","data_processing")

def load_data(path):
    try: 
        data=pd.read_csv(path)
        logger.debug("DATA LOAD SUCCESSFULLy")
        return data
    except Exception as e:
        logger.error("ERROR: ",e)
        raise

def processing_data(data):
    try:
        numcol=data.select_dtypes(include=['number']).columns.tolist()
        catcol=data.select_dtypes(include=['object']).columns.tolist()
        catpipe=Pipeline(steps=[("LABELENCODING",OrdinalEncoder())])
        numpipe=Pipeline(steps=[("SCALING",StandardScaler())])
        mergepipe=ColumnTransformer(transformers=[("CATEGORICAL_DATA",catpipe,catcol),("NUMERICAL_DATA",numpipe,numcol)],remainder="passthrough")
        finalpipe=Pipeline(steps=[('DATA_PROCESSING',mergepipe)])
        df=finalpipe.fit_transform(data)
        df1=pd.DataFrame(df,columns=catcol+numcol)
        temp=df1.pop("Test Results")
        df1.insert(len(df1.columns.tolist()),"Test Results",temp)
        logger.debug("DATA PROCESSING COMPLETED")
        return df1
    except Exception as e:
        logger.error("Error ",e)
        raise
def save_data(traindata,testdata,path):
    try:
        dirname="interim"
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
        trainpath="./process/raw/train.csv"
        testpath="./process/raw/test.csv"
        savepath="./process"
        traindata=load_data(trainpath)
        testdata=load_data(testpath)
        processtrain=processing_data(traindata)
        processtest=processing_data(testdata)
        save_data(processtrain,processtest,savepath)
        logger.debug("DATA PROCESSING STAGE COMPLETED ")
    except  Exception as e:
        logger.error("ERROR ",e)
        raise


if __name__ == "__main__":
    main()
