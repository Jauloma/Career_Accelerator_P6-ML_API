from fastapi import FastAPI # Import the FastAPI framework for building APIs
from typing import List, Literal # Import typing hints for function annotations
from pydantic import BaseModel # Import BaseModel for creating data models
import uvicorn # Import uvicorn for running the FastAPI app
import pandas as pd # Import pandas library for data manipulation
import pickle, os # Import pickle and os modules for handling files and data serialization

# Define a function to load machine learning components
def load_ml_components(fp):
    '''Load machine learning to re-use in app '''
    with open(fp, 'rb') as f:
        object = pickle.load(f) # Load a pickled object (machine learning model)
    return object # Return the loaded object

# Define a Pydantic model for the input data
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
    # Define the input features as class attributes

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
# Get the absolute path of the current directory
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Join the directory path with the model file name
ml_core_fp = os.path.join(DIRPATH, 'gradient_boosting_model.pkl')

# Define the labels manually
labels = ['Negative', 'Positive']

# Load the machine learning components
end2end_pipeline = load_ml_components(fp=ml_core_fp) # Load the machine learning model from the file

# Access the model step of the pipeline
model = end2end_pipeline.named_steps['model'] # Access the model component from the pipeline

# Create a dictionary to map index to labels
idx_to_labels = {i: l for (i, l) in enumerate(labels)}

# Print predictable labels and index-to-label mapping
print(f'\n[Info]Predictable labels: {labels}')
print(f'\n[Info]Indices to labels: {idx_to_labels}')

# Print information about the loaded model
print(f'\n[Info]ML components loaded - Model: {model}')

# Create the FastAPI application instance
app = FastAPI(title='Sepsis Prediction API') # Create a FastAPI instance with a title

# Define a route to handle the root endpoint
@app.get('/') 
def root():
    return{
        "info" : "Sepsis Prediction API: This interface is about the prediction of sepsis disease of patients in ICU."
    }
    

# Define a route to handle the prediction
@app.post('/classify')
def sepsis_classification(PlasmaGlucose : int,
    BloodWorkResult_1 : int,
    BloodPressure : int,
    BloodWorkResult_2 : int,
    BloodWorkResult_3 : int,
    BodyMassIndex : float,
    BloodWorkResult_4 : float,
    Age : int):
    # Define checkmarks for printing symbols
    red_x = u"\u274C"
    green_checkmark = "\033[32m" + u"\u2713" + "\033[0m" #u"\u2713"

    try:
         # # Create a dataframe from the input data, to solve the indexing issue, wrapp dict in a list
         df = pd.DataFrame(
            [ {
                'PlasmaGlucose': PlasmaGlucose,  
                'BloodWorkResult_1(U/ml)': BloodWorkResult_1,  
                'BloodPressure(mm Hg)': BloodPressure,  
                'BloodWorkResult_2(mm)': BloodWorkResult_2,  
                'BloodWorkResult_3(U/ml)': BloodWorkResult_3,  
                'BodyMassIndex(kg/m)^2': BodyMassIndex,  
                'BloodWorkResult_4(U/ml)':BloodWorkResult_4,  
                'Age (years)':Age}  ]
         )
         # Print input data as a dataframe
         print(f'[Info]Input data as dataframe:\n{df.to_markdown()}')

         # Predict using the loaded model
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
        
    # Create the API response
    result = {"Execution_msg": msg, "execution_code": code, "prediction": pred}
    return result

# Run the FastAPI application using uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000, reload = True)
