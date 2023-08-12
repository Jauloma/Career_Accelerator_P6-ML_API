# Sepsis Classification App Explained

## Importing Required Modules: 
In the beginning, the script imports essential modules that will be used throughout the program. The FastAPI module is imported from the FastAPI framework, which will be used to create the API. The List and Literal classes are imported from the typing module to define annotations for function arguments. The BaseModel class is imported from the pydantic module, allowing us to create data models with defined attributes. The uvicorn module is imported for running the FastAPI app, while the pandas module is imported for data manipulation. Finally, the pickle and os modules are imported for handling file operations and data serialization.

## Defining a Function to Load Machine Learning Components: 
A function named load_ml_components is defined to load machine learning components from a file. This function takes a file path as an argument, opens the file in binary read mode ('rb'), and uses the pickle.load() method to load the pickled object (machine learning model) from the file. The loaded object is returned to the caller.

## Defining the Pydantic Model for Input Data: 
A Pydantic data model named Sepsis is defined to represent the input data required for model prediction. This model is subclassed from the BaseModel class. It defines attributes that correspond to the features required for the model prediction, such as PlasmaGlucose, BloodWorkResult_1, BloodPressure, and so on. Additionally, the class attributes are documented using docstrings, explaining their purpose and data types.

## Setting Up the Application: 
This section deals with setting up the initial environment for the application. It first obtains the absolute path of the current directory using os.path.dirname(os.path.realpath(__file__)). The absolute path of the machine learning model file is then formed by joining the directory path with the filename (gradient_boosting_model.pkl). The labels list is defined manually to hold the possible target labels, i.e., 'Negative' and 'Positive'.

## Loading Machine Learning Components: 
Here, the script loads the machine learning components by calling the load_ml_components function with the previously obtained model file path (ml_core_fp). The loaded model is stored in the end2end_pipeline variable, and the actual machine learning model step is extracted and stored in the model variable using end2end_pipeline.named_steps['model'].

## Creating Index-to-Label Mapping: 
An index-to-label mapping is created using a dictionary comprehension. The dictionary, named idx_to_labels, maps the index to the corresponding label. This mapping will be useful later when converting predicted indices back to their corresponding labels.

## Printing Information About the Loaded Model: 
This part prints informative messages about the loaded model. The script prints the predictable labels extracted from the labels list, as well as the index-to-label mapping. It then prints information about the loaded model using the model variable, indicating that the machine learning components have been successfully loaded.

## Creating the FastAPI Application Instance: 
The FastAPI application instance is created using the FastAPI class. The title argument is set to 'Sepsis Prediction API', giving a title to the API.

## Defining the Root Endpoint: 
A route is defined using the @app.get('/') decorator, which handles the root endpoint. This route returns a JSON response containing information about the Sepsis Prediction API, explaining its purpose.

## Defining the Prediction Route: 
Another route is defined using the @app.post('/classify') decorator. This route handles predictions for sepsis classification. Inside the function sepsis_classification, symbols for checkmarks and a red "X" are defined using Unicode escape codes. The function processes the incoming data according to the defined Pydantic model and performs the following steps:

    1.	Creates a pandas DataFrame named df containing the input data attributes from the request.
    2.	Prints the input data in DataFrame format.
    3.	Uses the loaded machine learning model to predict the class labels for the input data. Additionally, it predicts the class probabilities using the predict_proba() method.
    4.	Retrieves the predicted class index and replaces it with the corresponding label in the DataFrame.
    5.	Maps the predicted indices to their corresponding labels.
    6.	Calculates and stores the confidence scores (probabilities) for each class in the DataFrame.
    7.	Prints the prediction results along with confidence scores as percentages.
    8.	Handles exceptions that may occur during the prediction process, printing error messages and returning a response indicating execution status.

## Running the FastAPI Application: 
Finally, the script checks if it's the main module and then runs the FastAPI application using uvicorn. The argument "main:app" specifies the module name and the FastAPI application instance. The reload argument is set to True to allow automatic reloading of the server.

This code represents a complete workflow for setting up a FastAPI-based API for sepsis prediction using a pre-trained machine learning model. It handles requests for classifying sepsis cases and provides informative responses, including predicted labels and confidence scores
