'''
Name:        Financial prediction NN using only leading indicators

Author:      Thomas Specq
Website:     http://www.thomas-specq.work
Link:        <a href="http://www.thomas-specq.work">Freelance Web Design & Développement</a>
Created:     24/08/2017
Copyright:   (c) Thomas Specq 2017
Licence:     BSD

'''

from __future__ import print_function
import numpy as np


import time
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
import talib
from sklearn.externals import joblib
from numpy import genfromtxt
import random, timeit
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

#Disable warning tf
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.set_printoptions(threshold=np.nan)

#Load data
def load_data():
    #Readour data from a csv file
    prices = pd.read_csv('lead.csv', delimiter=',')
    prices.rename(columns={'Closen': 'closen', 'Close':'close', 'Phousing':'phousing', 'Cci':'cci','IniClaim':'iniclaim','Money':'money','NewOrderCo':'neworderco','NewOrder':'neworder','WeekHour':'weekhour' }, inplace=True)

    x_train = prices.iloc[0:180]
    x_test= prices.iloc[180:210]
    return x_train, x_test, prices

#Initialize first state, all items are placed deterministically
#Every inputs except close and close next correspond to a leading indicator 
def init_state(indata):
    close = indata['close'].values
    closen = indata['closen'].values
    phousing = indata['phousing'].values
    cci = indata['cci'].values
    iniclaim = indata['iniclaim'].values
    money = indata['money'].values
    neworderco = indata['neworderco'].values
    neworder = indata['neworder'].values
    weekhour = indata['weekhour'].values
       
    #--- Preprocess data
    #xdata will be our inputs and ydata our outputs
    xdata = np.column_stack((close,phousing,cci,iniclaim,money,neworderco,neworder,weekhour))
    
    print(close.shape)
    time.sleep(5)

    variation = closen/close - 1
    #Our output is the variation of the stock price from a month to the other
    ydata = np.column_stack((variation))
   
    xdata = np.nan_to_num(xdata)
    ydata = np.nan_to_num(ydata)

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xdata)
    df_minmax = minmax_scale.transform(xdata)
    xdata = df_minmax

    ydata[ydata > 0] = 1
    ydata[ydata < 0] = 0
    ydata = np.transpose(ydata)
    return xdata, ydata


np.random.seed(0)  # for reproducibility

#Keras model initialization
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
from keras import backend as K

#8 inputs
num_features = 8

model = Sequential()
model.add(Dense(30, activation="sigmoid", kernel_initializer="lecun_uniform", input_dim=num_features))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation="sigmoid", kernel_initializer="lecun_uniform"))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid", kernel_initializer="lecun_uniform"))

#Compile the NN
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])

start_time = timeit.default_timer()

indataTrain, indataTest, prices = load_data()
normaX, normaY = init_state(indataTrain)
normaTestX, normaTestY = init_state(indataTest)

epochs = 10000
batch_size = 10000


model.fit(normaX, normaY, batch_size, epochs, verbose = 1)

scores = model.evaluate(normaTestX, normaTestY)

print()
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))


#ttime = datetime.now().strftime('%Y-%m-%d@%H_%M_%S')

#plt.figure()
#plt.suptitle('Summary'+str(ttime), fontsize=14, fontweight='bold')
#plt.plot(learning_progress)
#plt.show()











