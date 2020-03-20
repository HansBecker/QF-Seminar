# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:37:00 2020

@author: lodewijk
"""
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras import backend as K
import random as rn
import numpy as np
from numpy import genfromtxt
import pandas as pd


import csv

mydata = pd.read_csv("cleanedcalldata.csv")
calldata = mydata.to_numpy()
        
calldata = calldata[:,[0,3,4,5,6,7]]

floatdata = calldata.astype(np.float)


# np.random.seed(123)
# n_obs = 1_000
# n_reg = 5
  # n_obs x 5 matrix of regressors
X = floatdata[0:34000,1:]
Y = floatdata[0:34000,0]

########################################################
# --------------- fix the initial state ---------------#
# note: the optimization of neural networks involve random variables. To ensure
# your results are reproducible, always include this segment.
# see also https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)


tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
np.random.seed(123)

##########################################################
# --------------- do the actual modelling ---------------#

# Due to the way neural networks fit their parameters, you should always scale your X matrix!
offset = X.min(0)  # or X.mean(0)
scale = X.max(0) - offset  # or X.std(0)
X_scaled = (X - offset) / scale

_, k = X.shape

# create an empty network that we will build layer by layer
model = Sequential()

# first hidden layer, number of nodes is a multiple of the number of regressors (k)
model.add(Dense(4*k, input_dim=k, activation='relu'))

# second hidden layer, number of nodes can be toyed with
model.add(Dense(60, activation='relu'))

# the 'output layer', aka the y variable.
# the kernel regularizer is optional and performs variable selection.
# see https://en.wikipedia.org/wiki/Lasso_(statistics) for more info.
model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01)))

# convert your network description into fast machine code
model.compile(loss='mse', optimizer='adam')

print(model.summary())

# optimize the model parameters
history = model.fit(
    # do at most 1000 iterations of parameter optimization,
    epochs=1000,

    # the observations
    x=X_scaled,
    y=Y,

    # keep 20% of the data separate to determine the 'early stopping'
    validation_split=.2,
    callbacks=[
        # if the out-of-sample fit does not improve for 20 consecutive iterations, stop
        # optimizing and use the best model you have found so far.
        EarlyStopping(patience=20, restore_best_weights=True)
    ],
)

#################################################################################
# --------------- plot the simulated data and the network output ---------------#

fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144)

X_test = floatdata[34000:,1:]
Y_true = floatdata[34000:,0]
Y_pred = model.predict((X_test-offset)/scale)



# x = np.linspace(-2, 2, 20)
# x, y = np.meshgrid(x, x)

#make same float nr
Y_true = np.float32(Y_true)


#Get error predction 
Y_pred = Y_pred.ravel()
Y_error = np.subtract(Y_true, Y_pred)

# ax1.plot_surface(x, y, Y_true)
# ax1.set_title('The true data generating process')

# ax2.plot_surface(x, y, Y_pred)
# ax2.set_title('The network output')

# ax3.plot_surface(x, y, Y_pred-Y_true)
# ax3.set_title('The approximation error')

ax4.plot(history.history['loss'], label='in-sample mse')
ax4.plot(history.history['val_loss'], label='out-of-sample mse')
ax4.set_title('The mean squared error during optimization')
ax4.set_xlabel('no. of iterations')
ax4.set_ylabel('mean squared error')
ax4.legend()




