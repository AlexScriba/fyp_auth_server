import csv
import os

import numpy as np
import scipy as sp

from scipy import signal

from sklearn import *

import pickle

# Path to models dir
MODELS_PATH = './models'

# Path to actual models
MLP_MODEL_PATH = os.path.join(MODELS_PATH, 'mlp_model.pkl')
KM_MODEL_PATH = os.path.join(MODELS_PATH, 'km_model.pkl')
SCALER_PATH = os.path.join(MODELS_PATH, 'scaler.pkl')
TF_TRANSFORMER_PATH = os.path.join(MODELS_PATH, 'tf_transformer.pkl')

# Constants for data shape
NUM_DATA_POINTS = 7000
NUM_WORDS_IN_BOW = 80


# Load all models from their paths
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


mlp_model = load_model(MLP_MODEL_PATH)
km_model = load_model(KM_MODEL_PATH)
scaler = load_model(SCALER_PATH)
tf_tranformer = load_model(TF_TRANSFORMER_PATH)


# Function to interpolate data points using cubic interpolation
def interpolate(data, numAfter):
    timeStamps = np.array(list(map(lambda x: x[-1], data)))

    newTimeStamps = np.linspace(timeStamps.min(), timeStamps.max(), numAfter)

    res = []

    for i in range(len(data[0]) - 1):
        y = list(map(lambda x: x[i], data))

        new_y = sp.interpolate.interp1d(
            timeStamps, y, kind='cubic')(newTimeStamps)
        res.append(new_y)

    return np.array(res).transpose(1, 0)


# Function to combine the 4 data streams into one data vector
def combineData(requestData):
    accData = requestData['accelerometer']
    gravData = requestData['gravity']
    gyroData = requestData['gyroscope']
    rotData = requestData['rotation']

    accData = interpolate(accData, NUM_DATA_POINTS)
    gravData = interpolate(gravData, NUM_DATA_POINTS)
    gyroData = interpolate(gyroData, NUM_DATA_POINTS)
    rotData = interpolate(rotData, NUM_DATA_POINTS)

    retData = []

    for i in range(NUM_DATA_POINTS):
        retData.append(np.concatenate(
            [accData[i], gyroData[i], gravData[i], rotData[i]]))

    return np.array(retData)


# Function to transform input values to BOW encoding
def bow_transform(input):
    numwords = NUM_WORDS_IN_BOW
    w = km_model.predict(input)

    bw = np.bincount(w, minlength=numwords)
    return bw


# Function to apply low-pass Butterworth filter to data
def lowPassFilter(y):
    sos = signal.butter(10, 7, 'low', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, y)
    return filtered


# Function to handle pre-processing logic and prediction using models
def prediction(request_data):
    req_data = {}
    for key in request_data:
        req_data[key] = np.array(
            list(map(lambda x: np.float64(x), request_data[key])))

    combined_data = combineData(req_data)

    filtered_data = lowPassFilter(combined_data.transpose(1, 0)) \
        .transpose(1, 0)

    scaled_data = scaler.transform(filtered_data)

    bow = bow_transform(scaled_data)
    tfidf = tf_tranformer.transform([bow])[0]

    pred = mlp_model.predict(tfidf)[0]
    pred_proba = mlp_model.predict_proba(tfidf)[0]

    return (int(pred), pred_proba[0], pred_proba[1])
