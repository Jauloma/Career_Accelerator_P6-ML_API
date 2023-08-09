# Sepssis Disease prediction App

# Import required libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

# The creation of a FastAPI application instance.
# the 'app' instance is the main entry point for defining routes, 
# handling requests, and creating the API. 
app = FastAPI()


# The code defines a Pydantic model class named 'model_input' that 
# represents input data with specific fields for attributes related 
# to plasma glucose, blood work results, blood pressure, body mass index, 
# and age, each having specified data types.

class model_input(BaseModel):
    PlasmaGlucose : int
    BloodWorkResult_1 : int
    BloodPressure : int
    BloodWorkResult_2 : int
    BloodWorkResult_3 : int
    BodyMassIndex : float
    BloodWorkResult_4 : float
    Age : int

# Loading the save model
# rb='read binary'- opening the model in binary mode for reading

sepssis_model = pickle.load(open('gradient_boosting_model.pkl','rb')) 

#

@app.post('/sepssis_prediction')
def sepssis_pred(input_parameters : model_input):

    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)

    PRG = input_dictionary['PlasmaGlucose']
    PL = input_dictionary['BloodWorkResult_1']
    PR = input_dictionary['BloodPressure']
    SK = input_dictionary['BloodWorkResult_2']
    TS = input_dictionary['BloodWorkResult_3']
    M11 = input_dictionary['BodyMassIndex']
    BD2 = input_dictionary['BloodWorkResult_4']
    Age = input_dictionary['Age']

    input_list = [PRG,PL,PR,SK,TS,M11,BD2,Age]

    prediction = sepssis_model.predict([input_list])

    if prediction[0] == 0:
        return 'The patient in ICU has NO Sepssis disease'
    else:
        return 'The patient in ICU has Sepssis disease'
