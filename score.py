import joblib
from azureml.core.model import Model
import os
import json
import pandas as pd

def init():
    # Create a global variable called model
    global model
    # Load the model using the name of the model you registered
    model_path = Model.get_model_path("catboost_predictor")
    print("Model Path is  ", model_path)
    #Load the model from the path
    model = joblib.load(model_path)


def run(request):
   try:
     data = json.loads(request)
     df = pd.read_json(data)
     result = model.predict(df)
     return {'data' : result.tolist() , 'message' : "Successfully classified loan"}
   except Exception as e:
      error = str(e)
      return {'data' : error , 'message' : "Failed to classify loan"}

