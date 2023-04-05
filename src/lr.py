import csv
import os

import numpy as np
import scipy as sp

from sklearn import *

import pickle

MODELS_PATH = './models'

LR_MODEL_PATH = os.path.join(MODELS_PATH, 'lr_model.pkl')
KM_MODEL_PATH = os.path.join(MODELS_PATH, 'km_model.pkl')

# Normal: 7300, 100
NUM_DATA_POINTS = 7300
NUM_WORDS_IN_BOW = 100


with open(LR_MODEL_PATH, 'rb') as f:
    lr_model = pickle.load(f)

with open(KM_MODEL_PATH, 'rb') as f:
    km_model = pickle.load(f)


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


def bow_transform(input):
    numwords = NUM_WORDS_IN_BOW
    w = km_model.predict(input)

    bw = np.bincount(w, minlength=numwords)
    return bw


def prediction(requestData):
    reqData = {}
    for key in requestData:
        reqData[key] = np.array(
            list(map(lambda x: np.float64(x), requestData[key])))

    combinedData = combineData(reqData)
    bow = bow_transform(combinedData)
    pred = lr_model.predict([bow])[0]
    pred_proba = lr_model.predict_proba([bow])[0]
    # pred_proba = [0, 0]
    print(pred_proba)
    print(pred)

    return (int(pred), pred_proba[0], pred_proba[1])
