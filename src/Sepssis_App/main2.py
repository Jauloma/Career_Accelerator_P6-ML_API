from fastapi import FastAPI
from typing import List, Literal
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pickle, os

# Useful functions
def load_ml_components(fp):
    '''Load machine learning to re-use in app '''
    with open(fp, 'rb') as f:
        object = pickle.load(f)
    return object

# Input Modeling
class Sepsis(BaseModel):
    """
    Represents the input data for the model prediction.

    Attributes:
        PlasmaGlucose (int): The plasma glucose level of the individual.
        BloodWorkResult_1 (int): The result of blood work test 1.
        BloodPressure (int): The blood pressure reading of the individual.
        BloodWorkResult_2 (int): The result of blood work test 2.
        BloodWorkResult_3 (int): The result of blood work test 3.
        BodyMassIndex (float): The body mass index of the individual.
        BloodWorkResult_4 (float): The result of blood work test 4.
        Age (int): The age of the individual.

        'sepsis' is the target feature which holds 0 = Negative and 1 = Positive.
    """
    # Here are the input features:

    PlasmaGlucose : int
    BloodWorkResult_1 : int
    BloodPressure : int
    BloodWorkResult_2 : int
    BloodWorkResult_3 : int
    BodyMassIndex : float
    BloodWorkResult_4 : float
    Age : int

# Setup
"""
Get the absolute path of the current model file.
then extracts the directory path from the absolute path of the model file.
This is useful when we need to locate the file 
relative to our script's location.
"""
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'gradient_boosting_model.pkl')

# Define the labels manually
labels = ['Negative', 'Positive']

# Loading
ml_components_dict = load_ml_components(fp=ml_core_fp)

idx_to_labels = {i: l for (i, l) in enumerate(labels)}
end2end_pipeline = ml_components_dict

print(f'\n[Info]Predictable labels: {labels}')
print(f'\n[Info]Indices to labels: {idx_to_labels}')
print(f'\n[Info]ML components loaded: {list(ml_components_dict.keys())}')

# API
app = FastAPI(title='Sepsis Prediction API')

@app.get('/')
async def root():
    return{
        "info": "Sepsis Prediction API: This interface is about the prediction of sepsis disease of patients in ICU."
    }

@app.post('/classify')
async def sepsis_classification(sepsis: Sepsis):
    try:
        # Create dataframe
        df = pd.DataFrame(
            {
            'PlasmaGlucose': [sepsis.PRG],
            'BloodWorkResult_1 (mu U/ml)': [sepsis.PL],
            'BloodPressure (mm Hg)': [sepsis.PR],
            'BloodWorkResult_2 (mm)': [sepsis.SK],
            'BloodWorkResult_3  (mu U/ml)': [sepsis.TS],
            'BodyMassIndex (weight in kg/(height in m)^2': [sepsis.M11],
            'BloodWorkResult_4 (mu U/ml)': [sepsis.BD2],
            'Age (years)': [sepsis.Age]}
        )
        print(f'[Info]Input data as dataframe:\n{df.to_markdown()}')

        # ML part
        output = end2end_pipeline.predict_proba(df)
        # store confidence score/probability for the predicted classification
        confidence_score = output.max(axis = -1)
        df['Confidence score'] = confidence_score

        ## Get index of predicted class
        predicted_idx = output.argmax(axis = -1)

        # Store index then replace by the matching label
        df['Predicted label'] = predicted_idx
        predicted_label = df['Predicted label'].replace(idx_to_labels)
        df['Predicted label'] = predicted_label
        
        # Define a Unicode character for a checkmarks
        green_checkmark = u"\u2713"
        red_x = u"\u274C"

        print(f"{green_checkmark} This patient in ICU has been classified as: {predicted_label}")
        msg = "Execution went fine"
        code = 1
        pred = df.to_dict("records")
    except:
        print(f"\033[91m{red_x} Something went wrong during the prediction of patient's sepsis state")
        msg = "Execution did not go well"
        code = 0
        pred = None
    result = {"Execution_msg": msg, "execution_code":code, "prediction" : pred}
    return result

        

