#test hyperparameter tuning

#import modules
import time

import numpy as np
import pandas as pd
from pandas import ExcelWriter, ExcelFile
import xlsx_utils as xlsx
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import preprocessing
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, initializers
from keras.wrappers.scikit_learn import KerasRegressor

#read dataset
x_data, y_data, header = xlsx.read('6sample.xlsx')

#kFold
kf = KFold(3, shuffle=True)
fold = 0

#create NN model
def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=3, activation='tanh'))
    model.add(Dense(2, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

for train, test in kf.split(x_data):
    fold += 1

    x_train = x_data[train]
    y_train = y_data[train]

    x_test = x_data[test]
    y_test = y_data[test]

    print("Fold #{}: train={}, test={}".format(fold, train, test))

    #data standardization
    std_scale_x = preprocessing.StandardScaler().fit(x_train)
    std_scale_y = preprocessing.StandardScaler().fit(y_train)

    x_scale_train = std_scale_x.transform(x_train)
    x_scale_test = std_scale_x.transform(x_test)

    y_scale_train = std_scale_y.transform(y_train)
    y_scale_test = std_scale_y.transform(y_test)

    #data normalization
    normalized = MinMaxScaler()
    normalized.fit(x_scale_train)

    xtrain = normalized.transform(x_scale_train)
    xtest = normalized.transform(x_scale_test)

    norm = MinMaxScaler()
    norm.fit(y_scale_train)

    ytrain = norm.transform(y_scale_train)
    ytest = norm.transform(y_scale_test)

    validator = KerasRegressor(build_fn=create_model, epochs=1000, verbose=0)

    #define the grid search parameter
    optimizer = ['Adagrad', 'Adam']
    param_grid = dict(optimizer=optimizer)

    NNmodel = GridSearchCV(estimator=validator, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=None)
    NNmodel.fit(xtrain, ytrain, batch_size=None)
