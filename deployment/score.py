import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
from azureml.core import Model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def init():
    global model, scaler
    model_dir = Model.get_model_path('nasa-bearing-anomaly-prediction')
    scaler = joblib.load(os.path.join(model_dir, 'scaler_data'))
    model = load_model(os.path.join(model_dir, 'model.h5'))

def run(raw_data):
    
    # Read and preprocess data
    test = pd.read_json(json.loads(raw_data)['test_data'])
    X_test = scaler.transform(test)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Calculate the loss based on input data
    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.15
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    
    return json.dumps({'result': scored.to_json()}) 
