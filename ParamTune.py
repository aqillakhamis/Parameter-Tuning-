#test hyperparameter tuning

#import modules
import time

import numpy as np
import pandas as pd
from pandas import ExcelWriter, ExcelFile
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import preprocessing
from sklearn import metrics 
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import xlsx_utils as xlsx
import warnings 
warnings.filterwarnings("ignore")

#read dataset 
Data, target, header = xlsx.read('6sample.xlsx')

D_train, D_test, t_train, t_test = train_test_split(Data, target, test_size=0.5)

#feature scaling 
#data standardization 
std_scale_x = preprocessing.StandardScaler().fit(D_train)
std_scale_y = preprocessing.StandardScaler().fit(t_train)

D_train = std_scale_x.transform(D_train)
D_test = std_scale_x.transform(D_test)

t_train = std_scale_y.transform(t_train)
t_test = std_scale_y.transform(t_test)

#data normalization 
normalized = MinMaxScaler()
normalized.fit(D_train)

D_train = normalized.transform(D_train)
D_test = normalized.transform(D_test)

norm = MinMaxScaler()
norm.fit(t_train)

t_train = norm.transform(t_train)
t_test = norm.transform(t_test) 

#kFold
kf = KFold(n_splits=3)

class MyKerasRegressor(KerasRegressor):
    """Implementation of the scikit-learn regressor API for Keras.
    """

    def predict(self, x, **kwargs):
        """Returns predictions for the given test data.
        Notes
        -----
        This is a fix for KerasRegressor.
        Replaced ``return np.squeeze(self.model.predict(x, **kwargs))`` with
        ``return self.model.predict(x, **kwargs)``.
        The np.squeeze() causes shape inconsistent. For example, with np.squeeze() the output shape
        become (2,) if input shape is (1,2). The output shape should be the same as input shape.
        Arguments:
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict`.
        Returns:
            preds: array-like, shape `(n_samples,)`
                Predictions.
        """
        kwargs = self.filter_sk_params(Model.predict, kwargs)
        #return np.squeeze(self.model.predict(x, **kwargs))
        return self.model.predict(x, **kwargs)
    
#create NN model
def create_model(optimizer='adam'):
    # create model   
    #random normal distribution with fixed seed number for random number generator 
    random_normal = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0)
    
    #input layer 
    input_layer = Input(shape=(D_train.shape[1],), name="input")
    
    #hidden layer 
    hidden_layers = input_layer
    for i in range(1): 
        hidden_layers = Dense(10,
                              activation='tanh',
                              kernel_initializer=random_normal,
                              bias_initializer=initializers.Constant(value=1.0),
                              name="hidden_%d" % (i+1))(hidden_layers)
        
    #output layer
    output_layer = Dense(t_train.shape[1],
                         activation='linear',
                         kernel_initializer=random_normal,
                         name="output")(hidden_layers) 

    model = Model(input_layer, output_layer)
    model.compile(loss='mean_squared_error', optimizer=optimizer)    
    return model 

validator = MyKerasRegressor(build_fn=create_model, epochs=10000, verbose=0)

#define the grid search parameter
optimizer = ['Adagrad','Adam','Adadelta','SGD','RMSprop']
param_grid = dict(optimizer=optimizer)

NNmodel = GridSearchCV(estimator=validator, param_grid=param_grid, n_jobs=-1, cv=kf, verbose=0)
NNresult = NNmodel.fit(D_train,t_train, batch_size=None)

# summarize results
print("Best: %f using %s" % (-NNresult.best_score_, NNresult.best_params_))
means = -NNresult.cv_results_['mean_test_score']
stds = NNresult.cv_results_['std_test_score']
params = NNresult.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##############################################################
    


