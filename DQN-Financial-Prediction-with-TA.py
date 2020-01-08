'''
Name:        Financial prediciton with DQN using most common Technical indicators
	     based on the work from Daniel Zakrisson
             https://hackernoon.com/the-self-learning-quant-d3329fcc9915
             
Author:      Thomas Specq
Website:     http://www.thomas-specq.work
Link:        <a href="http://www.thomas-specq.work">Freelance Web Design & Developpement</a>
Created:     24/08/2017
Copyright:   (c) Thomas Specq 2017 
	     parts of the DQN codes belonging to Daniel Zakrisson
Licence:     BSD

'''

from __future__ import print_function

import numpy as np
np.random.seed(0)  # for reproducibility
np.set_printoptions(threshold=np.nan)
import time
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
import talib
from sklearn.externals import joblib
from numpy import genfromtxt
from datetime import datetime

#Disable warning tf
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Load data
def load_data(test=False):

    prices = pd.read_csv('aapl.csv', delimiter=',')
    prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Close1':'close1', 'Close2':'close2', 'Close3':'close3', 'Close4':'close4', 'Close5':'close5', 'Close6':'close6', 'Close7':'close7', 'Close8':'close8', 'Close9':'close9','Close10':'close10','Close11':'close11','Close12':'close12','Close13':'close13','Close14':'close14'}, inplace=True)

    x_train = prices.iloc[0:3400]
    x_test= prices.iloc[3400:4270]

    if test:
        return x_test
    else:
        return x_train

#Initialize first state, all items are placed deterministically
#Every inputs except close correspond to a technical indicator 
def init_state(indata, test=False):
    openn = indata['open'].values
    close = indata['close'].values
    high = indata['high'].values
    low = indata['low'].values
    volume = indata['volume'].values
    
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    
    sma30 = talib.SMA(close, 30)
    sma60 = talib.SMA(close, timeperiod=60)
    rsi = talib.RSI(close, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)
    trange = talib.TRANGE(high, low, close)
    macd, macdsignal, macdhist = talib.MACD(close, 12, 26, 9)
    upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
    ema = talib.EMA(close, 30)
    ma = talib.MA(close, 30)
    wma = talib.WMA(close, timeperiod=30)
    tema = talib.TEMA(close, 30)
    obv = talib.OBV(close, np.asarray(volume, dtype='float'))
    adx = talib.ADX(high, low, close, 14)
    apo = talib.APO(close, 12, 2, 0)
    bop = talib.BOP(openn, high, low, close)
    mom = talib.MOM(close,10)
    ppo = talib.PPO(close, 12, 26, 0)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    ad = talib.AD(high, low, close, np.asarray(volume, dtype='float'))
    wcl = talib.WCLPRICE(high, low, close)

    #--- Preprocess data
    xdata = np.column_stack((close, diff, sma30, sma60, rsi, atr, macd, macdsignal, macdhist, lower, middle, upper, ema, ma, wma, adx, apo, bop, mom, ppo, slowk, slowd, trange, wcl))
    
    xdata = np.nan_to_num(xdata)
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    state = xdata[0:1, 0:1, :]

    return state, xdata, close

#Take Action
def take_action(state, xdata, action, signal, time_step):
    global position
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett
    
    #make necessary adjustments to state and then return it
    time_step += 1
    
    #if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        state = xdata[time_step-1:time_step, 0:1, :]
        terminal_state = 1
        signal.loc[time_step] = 0

        return state, time_step, signal, terminal_state

    #move the market data window one step forward
    state = xdata[time_step-1:time_step, 0:1, :]
    #take action
    if (action == 1) :
        signal.loc[time_step] = 100
        position = signal.loc[time_step]

    elif (action == 2) :
        signal.loc[time_step] = -100
        position = signal.loc[time_step]
    else:
        signal.loc[time_step] = 0

    terminal_state = 0
    return state, time_step, signal, terminal_state

#Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    if eval == False:
        bt = twp.Backtest(pd.Series(data=[x for x in xdata[time_step-2:time_step]], index=signal[time_step-2:time_step].index.values), signal[time_step-2:time_step], signalType='shares')
        reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

    if terminal_state == 1 and eval == True:
        #save a figure of the test set
        bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]

    #print(time_step, terminal_state, eval, reward)

    return reward

def evaluate_Q(eval_data, eval_model, price_data, epoch=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True, epoch=epoch)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0

    return eval_reward

#This neural network is the the Q-function, run it like this:
#model.predict(state.reshape(1,64), batch_size=1)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

tsteps = 1
batch_size = 30
num_features = 24

model = Sequential()
model.add(Dense(500, activation="sigmoid", kernel_initializer="lecun_uniform", input_shape=(1,num_features)))
#model.add(Activation('relu'))
model.add(Dropout(0.2)) 

model.add(Dense(500, activation="sigmoid", kernel_initializer="lecun_uniform"))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(4, activation="linear", kernel_initializer="lecun_uniform"))
#model.add(Activation('linear')) #linear  output so we can have range of real-valued outputs

rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer='adam')

import random, timeit
start_time = timeit.default_timer()

indata = load_data()
test_data = load_data(test=True)
epochs = 1
gamma = 0.99 #since the reward can be several time steps away, make gamma high
epsilon = 1
batchSize = 200
buffer = 200
replay = []
learning_progress = []

position = 0
#stores tuples of (S, A, R, S')
h = 0
#signal = pd.Series(index=market_data.index)
signal = pd.Series(index=np.arange(len(indata)))
for i in range(epochs):
    if i == epochs-1: #the last epoch, use test data set
        indata = load_data(test=True)
        state, xdata, price_data = init_state(indata, test=True)
        print(xdata)
        print()
        print(price_data)
    else:
        state, xdata, price_data = init_state(indata)
    status = 1
    terminal_state = 0
    #time_step = market_data.index[0] + 64 #when using market_data
    time_step = 14
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions

        qval = model.predict(state, batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4) #assumes 4 different actions
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
            #print(time_step, reward, terminal_state)
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state, batch_size=1)
                newQ = model.predict(new_state, batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                if terminal_state == 0: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                #print(time_step, reward, terminal_state)
                X_train.append(old_state)
                y_train.append(y.reshape(4,))

            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
            
            state = new_state
        if terminal_state == 1: #if reached terminal state, update epoch status
            status = 0
    eval_reward = evaluate_Q(test_data, model, price_data, i)
    learning_progress.append((eval_reward))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Epoch #: %s Reward: %f Epsilon: %f" % (i,eval_reward, epsilon))
    #learning_progress.append((reward))
    epsilon = 100-(np.log(i+1)*0.1)*50
    epsilon = np.around(epsilon/100,2)
    if(i >= 300):
    	decrease = 0.0023*i
	epsilon = 1.4 - decrease 
    if(epsilon <= 0.1):
	epsilon = 0.1


elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

bt = twp.Backtest(pd.Series(data=[x[0,0] for x in xdata]), signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)
print(bt.data)

lengthBT = len(bt.data) - 1
lastpnl = round(bt.data['pnl'][lengthBT],2)

unique, counts = np.unique(filter(lambda v: v==v, signal.values), return_counts=True)
print(np.asarray((unique, counts)).T)


ttime = datetime.now().strftime('%Y-%m-%d@%H_%M_%S')

plt.figure()
plt.suptitle('P&L :'+str(lastpnl), fontsize=14, fontweight='bold')
plt.subplot(3,1,1)
bt.plotTrades()
plt.subplot(3,1,2)
bt.pnl.plot(style='x-')
plt.subplot(3,1,3)
plt.plot(learning_progress)
plt.savefig('plt/'+ttime+'.png', bbox_inches='tight', pad_inches=10, dpi=600)
plt.show()










