import pytest
import joblib as jb
import pandas as pd
import sklearn

Model=jb.load("./models/model.pkl")
featurepipe=jb.load("./processpipe/feature_pipe.pkl")
targetpipe=jb.load("./processpipe/target_pipe.pkl")
features=Model.feature_names_in_.tolist()

df1=pd.DataFrame([{
            "Age": 19,
            "Gender": "Male",
            "Blood Type": "B+",
            "Medical Condition": "Diabetes",
            "Billing Amount": 1234.222,
            "Room Number": 321,
            "Medication": "Aspirin"
        }])
df2 = pd.DataFrame([{
    "Age": 57,
    "Gender": "Male",
    "Blood Type": "O+",
    "Medical Condition": "Diabetes",
    "Billing Amount": 3616.89,
    "Room Number": 339,
    "Medication": "Aspirin",
}])

df3 = pd.DataFrame([{
    "Age": 51,
    "Gender": "Female",
    "Blood Type": "A+",
    "Medical Condition": "Diabetes",
    "Billing Amount": 36970.08,
    "Room Number": 372,
    "Medication": "Penicillin",
}])

df4 = pd.DataFrame([{
    "Age": 20,
    "Gender": "Male",
    "Blood Type": "A+",
    "Medical Condition": "Asthma",
    "Billing Amount": 44393.00,
    "Room Number": 148,
    "Medication": "Penicillin",
}])

df5 = pd.DataFrame([{
    "Age": 74,
    "Gender": "Female",
    "Blood Type": "B+",
    "Medical Condition": "Obesity",
    "Billing Amount": 27554.92,
    "Room Number": 135,
    "Medication": "Ibuprofen",
}])
def predict(rawdata,featurepipe,targetpipe,Model):
        processdata=featurepipe.transform(rawdata)
        processdata=pd.DataFrame(processdata,columns=rawdata.columns.tolist())
        processdata=processdata[features]
        rawouput=Model.predict(processdata)
        output=targetpipe.inverse_transform([rawouput])
        return output[0][0]

def test1():
    value=predict(df1,featurepipe,targetpipe,Model)
    assert value in targetpipe.named_steps['LABELENCODING'].categories_[0].tolist(), f"Unexpected class {value}"
def test2():
    value=predict(df2,featurepipe,targetpipe,Model)
    assert value in targetpipe.named_steps['LABELENCODING'].categories_[0].tolist(), f"Unexpected class {value}"
def test3():
    value=predict(df3,featurepipe,targetpipe,Model)
    assert value in targetpipe.named_steps['LABELENCODING'].categories_[0].tolist(), f"Unexpected class {value}"
def test4():
    value=predict(df4,featurepipe,targetpipe,Model)
    assert value in targetpipe.named_steps['LABELENCODING'].categories_[0].tolist(), f"Unexpected class {value}"
def test5():
    value=predict(df5,featurepipe,targetpipe,Model)
    assert value in targetpipe.named_steps['LABELENCODING'].categories_[0].tolist(), f"Unexpected class {value}"    
