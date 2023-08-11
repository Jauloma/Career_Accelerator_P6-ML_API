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
end2end_pipeline = load_ml_components(fp=ml_core_fp)

# Access the model step of the pipeline
model = end2end_pipeline.named_steps['model']

idx_to_labels = {i: l for (i, l) in enumerate(labels)}

print(f'\n[Info]Predictable labels: {labels}')
print(f'\n[Info]Indices to labels: {idx_to_labels}')
# Print information about the loaded model
print(f'\n[Info]ML components loaded - Model: {model}')

# API
app = FastAPI(title='Sepsis Prediction API')

@app.get('/')
async def root():
    return{
        "info": "Sepsis Prediction API: This interface is about the prediction of sepsis disease of patients in ICU."
    }

@app.post('/classify')
async def sepsis_classification(sepsis: Sepsis):
    # Define checkmarks
    red_x = u"\u274C"
    green_checkmark = "\033[32m" + u"\u2713" + "\033[0m" #u"\u2713"

    try:
         # Create dataframe
         df = pd.DataFrame(
             {
                'PlasmaGlucose': [sepsis.PlasmaGlucose],  
                'BloodWorkResult_1(U/ml)': [sepsis.BloodWorkResult_1],  
                'BloodPressure(mm Hg)': [sepsis.BloodPressure],  
                'BloodWorkResult_2(mm)': [sepsis.BloodWorkResult_2],  
                'BloodWorkResult_3(U/ml)': [sepsis.BloodWorkResult_3],  
                'BodyMassIndex(kg/m)^2': [sepsis.BodyMassIndex],  
                'BloodWorkResult_4(U/ml)': [sepsis.BloodWorkResult_4],  
                'Age (years)': [sepsis.Age]}  
         )
         print(f'[Info]Input data as dataframe:\n{df.to_markdown()}')

         # ML part
         output = model.predict(df)
         confidence_scores = model.predict_proba(df)  # Predict the probabilities for each class
         print(f'Considering the best confidence score, the output is: {output}')
         print(f'Confidence scores: {confidence_scores}')

         # Get index of predicted class
         predicted_idx = output

         # Store index then replace by the matching label
         df['Predicted label'] = predicted_idx
         predicted_label = df['Predicted label'].replace(idx_to_labels)
         df['Predicted label'] = predicted_label

         # Map predicted indices to labels
         predicted_labels = [idx_to_labels[idx] for idx in output]

         # Store the predicted probabilities for each class in the dataframe
         for i, label in enumerate(labels):
             df[f'Confidence_{label}'] = confidence_scores[:, i] * 100  # Convert to percentage

             # Print the result with confidence scores as percentages
             if predicted_labels:
                  i = 0  
                  label = predicted_labels[0]  # Get the first predicted label
                  confidence_score_percentage = max(confidence_scores[i]) * 100
                  print(f"{green_checkmark} This patient in ICU has been classified as Sepsis {label} with confidence of: {confidence_score_percentage:.1f}%")

         msg = "Execution went fine"
         code = 1
         pred = df.to_dict("records") 
         

    except Exception as e:
        print(f"\033[91m{red_x} An exception occurred: {str(e)}")
        msg = "Execution did not go well"
        code = 0
        pred = None
        
    result = {"Execution_msg": msg, "execution_code": code, "prediction": pred}
    return result
if __name__ == "__main__":
    uvicorn.run("main2:app", reload = False)
