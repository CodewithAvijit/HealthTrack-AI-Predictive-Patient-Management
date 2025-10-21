from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,field_validator
from typing import Literal,Annotated
import joblib as jb
import pandas as pd
import sklearn
from fastapi.middleware.cors import CORSMiddleware

MODEL_VERSION='v1.0.1'
Model=jb.load("./models/model.pkl")
featurepipe=jb.load("./processpipe/feature_pipe.pkl")
targetpipe=jb.load("./processpipe/target_pipe.pkl")
features=Model.feature_names_in_.tolist()

app=FastAPI(
    version=MODEL_VERSION,  
    title="AI HEALTH CHECKER",
    description=f"THIS MODEL MADE USING {type(Model).__name__} Algorithm to predict Health"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],        # Origins allowed
    allow_credentials=True,       # Allow cookies, headers
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)
class Inputs(BaseModel):
    Age: Annotated[int,Field(description="AGE OF PATIENT",example=19)]
    Gender: Annotated[Literal['Male', 'Female'],Field(description="GENDER OF THE PATIENT",example='Male')]
    Blood: Annotated[Literal['B-', 'A+', 'A-', 'O+', 'AB+', 'AB-', 'B+', 'O-'],Field(description="Blood group of patient",example='B-')]
    condition: Annotated[Literal['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis'],Field(description="MEDICAL CONDITION OF THE PATIENT",example='Asthma')]
    Amount: Annotated[float,Field(description="BILLING AMOUNT OF THE PATIENT",example=17695.911622343818)]
    Room: Annotated[int,Field(description="ROOM number of patient",example=345)]
    Medication: Annotated[Literal['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin', 'Lipitor'],Field(description="Medicine of the patient",example='Penicillin')]
    @field_validator('Age')
    @classmethod
    def type(cls,value):
        value=int(value)
        if(value<=0 ):
            raise ValueError("USE MORE THAN 0")
        else:
            return
    @field_validator('Amount')
    @classmethod
    def amount(cls,value):
        value=float(value)
        if(value<=0.0):
            raise ValueError("USE MORE THAN 0")
        else:
            return value

@app.get('/',tags=['About'])
def about():
    try:
        return JSONResponse(content={"AI HEALTH PREDICTOR":"MACHINE LEARNING MDOEL"},status_code=200)
    except Exception as e:
        return JSONResponse(content=str(e))
    
@app.get('/health',tags=['Health'])
def health():
    try:
        return JSONResponse(content=
                            {
                                "Model":True if Model else False,
                                "Model Name ": type(Model).__name__,
                                "Internal Features ":features,
                                "Featurepipe ":True if featurepipe else False,
                                "Targetpipe ":True if targetpipe else False
                             }
                            )
    except Exception as e:
        return JSONResponse(content=str(e))


@app.post('/predict')
def predict(data:Inputs):
    try:
        rawdata= {
            "Age": data.Age,
            "Gender": data.Gender,
            "Blood Type": data.Blood,
            "Medical Condition": data.condition,
            "Billing Amount": data.Amount,
            "Room Number": data.Room,
            "Medication": data.Medication
        }
        rawdata=pd.DataFrame([rawdata])
        processdata=featurepipe.transform(rawdata)
        processdata=pd.DataFrame(processdata,columns=rawdata.columns.tolist())
        processdata=processdata[features]
        rawouput=Model.predict(processdata)
        output=targetpipe.inverse_transform([rawouput])
        return JSONResponse(content={"HEALTH: ":output[0][0]},status_code=200)
    except Exception as e:
        return JSONResponse(content=str(e))



