import random
import numpy as np
import tensorflow as tf

seed_value= 1111
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

import warnings
import sys
import os
import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Input,LSTM, Dense, Flatten, Conv1D, Lambda, Reshape
from keras.layers.merge import concatenate, multiply,add
from keras import regularizers
from keras.initializers import glorot_uniform
from tqdm import tqdm
from keras import regularizers
from keras.models import load_model
#from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
########################################################################################
########################################################################################
#Reading Datasets

# Main Dataset
data= pd.read_csv("Data/cif_dataset_complete.csv",header=None)
# Expert Knowledge
predictions = pd.read_csv("Data/cif_theta_25_horg.csv",index_col=0,skiprows = [1])
print(data.shape)
print(predictions.shape)
########################################################################################
########################################################################################
# Functions to Make Inputs

def make_input(data,window_size,horizon=1):
    length=data.shape[0]
#     depth=data.shape[2]
    y = np.zeros([length-window_size+1-horizon,horizon])
    output=np.zeros([length-window_size+1-horizon,window_size])
    for i in range(length-window_size+1-horizon):
        output[i:i+1,:]=data[i:i+window_size]
        y[i,:]= data[i+window_size:i+window_size+horizon]
    return output.reshape(output.shape[0],window_size,1), y
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def make_k_input(data,window_size,horizon):
    length = data.shape[0]
    output= np.zeros([length-window_size+1-horizon,horizon])
    for i in range(length-window_size-horizon+1):
        output[i:i+1,:]=data[i+window_size:i+window_size+horizon]
    return output.reshape(output.shape[0],horizon,1)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def nonov_make_input(data,window_size,horizon=1):
    length=data.shape[0]-window_size
    loop=length//horizon
    extra = length%horizon
#     print(str(extra))
    data = np.append(data,np.zeros([horizon-extra]))
#     print(data)
    if extra ==0:
        i_val = loop
    else:
        i_val=loop+1
        
    output=np.zeros([i_val,window_size])
    y=np.zeros([i_val,horizon])
    for i in range(i_val):
        output[i:i+1,:]=data[i*horizon:(i*horizon)+window_size]
        y[i,:]= data[(i*horizon)+window_size:(i*horizon)+window_size+horizon]
        
    return output.reshape(output.shape[0],window_size,1), y
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def nonov_make_k_input(data,window_size,horizon):
    length = data.shape[0]-window_size
    loop=length//horizon
    extra = length%horizon
    data_app = np.repeat(data[-1],extra)
    data = np.append(data,data_app)    
#     data = np.append(data,np.zeros([horizon-extra]))
    if extra ==0:
        i_val = loop
    else:
        i_val=loop+1
    output=np.zeros([i_val,horizon])
    for i in range(i_val):
        output[i:i+1,:]=data[(i*horizon)+window_size:(i*horizon)+window_size+horizon]
    return output.reshape(output.shape[0],horizon,1)    

########################################################################################
########################################################################################
# Evaluation Function (SMAPE)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200.0 * np.mean(diff)

########################################################################################
########################################################################################
#Main Code

#warnings.filterwarnings('ignore')
data_length = data.shape[0]
with tqdm(total=data_length) as pbar:
    final_predictions = np.zeros([data_length,12])
    final_test = np.zeros([data_length,12])
    final_smape=np.zeros(data_length)
    for y in range(data_length):
        num_test = data.iloc[y].values[1]
        horizon=6
        window_size=6
        current_row =np.asarray(data.loc[y][3:].dropna().values,dtype=float)
        rr = current_row.size
        rr = int(np.floor(rr*.25))
        series_d=current_row[rr:] # values after (rr)th position in current_row
        series_data = series_d[:-num_test]
        series_length = series_data.size
        current_test=series_d[-num_test:]
        n_val = int(np.round(series_length*.2))
        if n_val < horizon: 
            n_val = horizon            
        LL=series_data.shape[0]-n_val
        HH=LL-window_size+1-horizon
        if (HH<0):
            train = series_data[:-n_val+1-HH]
        else:
            train = series_data[:-n_val]                        

        test = series_d[-(num_test+window_size):]
        val = series_data[-(n_val+window_size):]

        train_sequence = make_input(train, window_size,horizon)
        val_sequence = make_input(val,window_size,horizon)
        test_sequence = nonov_make_input(test,window_size,horizon)

        train_sequence_norm = deepcopy(train_sequence)
        val_sequence_norm = deepcopy(val_sequence)
        test_sequence_norm = deepcopy(test_sequence)
        
        scaler = MinMaxScaler()
        for j in range(train_sequence[0].shape[0]):
            scaler.fit(train_sequence[0][j])
            train_sequence_norm[0][j]= scaler.transform(train_sequence[0][j])
            train_sequence_norm[1][j]= scaler.transform(train_sequence[1][j].reshape(train_sequence[1][j].shape[0],1)).reshape(train_sequence[1][j].shape[0])    
        
        for j in range(val_sequence[0].shape[0]):
            scaler.fit(val_sequence[0][j])
            val_sequence_norm[0][j]= scaler.transform(val_sequence[0][j])
            val_sequence_norm[1][j]= scaler.transform(val_sequence[1][j].reshape(val_sequence[1][j].shape[0],1)).reshape(val_sequence[1][j].shape[0])

        for j in range(test_sequence[0].shape[0]):
            scaler.fit(test_sequence[0][j])
            test_sequence_norm[0][j]= scaler.transform(test_sequence[0][j])
            test_sequence_norm[1][j]= scaler.transform(test_sequence[1][j].reshape(test_sequence[1][j].shape[0],1)).reshape(test_sequence[1][j].shape[0])

        x_train = train_sequence_norm[0]
        y_train =train_sequence_norm[1]
        x_val = val_sequence_norm[0]
        y_val = val_sequence_norm[1]        
        x_test = test_sequence_norm[0]
        y_test = test_sequence_norm[1]
        
        train_input = x_train
        val_input = x_val
        test_input = x_test
####################################################################################
################       Neural Network Configuration         ########################
####################################################################################
        tf.reset_default_graph()
        K.clear_session()
        
        input_data= Input(batch_shape=(None,window_size,1),name='input_data')
        
        branch_0 = Conv1D(32,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(input_data)
        branch_1 = Conv1D(32,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(branch_0)
        branch_2 = Conv1D(32,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(branch_1)
        branch_3=Flatten()(branch_2)
        net= Dense(horizon,name='dense_final',activity_regularizer=regularizers.l2(0.0001))(branch_3)        

        model=Model(inputs=[input_data],outputs=net)

        callback = ModelCheckpoint(filepath='CIF_NetworkWeights/checkpoint_CIF_01_%i.h5' %y,monitor='val_loss',save_best_only=True,save_weights_only=True)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
        model.fit({'input_data':train_input},y_train,validation_data=[[val_input],y_val],callbacks=[callback],batch_size=8,shuffle=True, epochs=100,verbose=0)
        model.load_weights('CIF_NetworkWeights/checkpoint_CIF_01_%i.h5' %y)
#######################################################################################################
        if (num_test==12):
            test_input_temp=np.zeros([1,window_size,1])
            test_input_temp[0]=test_input[0]
            pred_temp=model.predict({'input_data':test_input_temp})
            for z in range(horizon):
                test_input[1][z]=pred_temp[0][z]
#######################################################################################################        
        pred=model.predict({'input_data':test_input})
        pred=scaler.inverse_transform(pred)

        final_predictions[y,:num_test] = pred.reshape(num_test)
        final_test[y,:num_test]=test[-num_test:]

        AAA=smape(pred.reshape(num_test),current_test)
        final_smape[y]=AAA
        pbar.update(1)

np.savetxt('Output/CIF_Prediction_01.csv',final_predictions, fmt='%1.3f',delimiter=',')
model.summary()
print('-----------------')
print("SMAPE:")
print (sum(final_smape)/data_length)    



#######################################################################################################
#######################################################################################################
#######################################################################################################

################################################# RESULTS #############################################

#######################################################################################################
#######################################################################################################
#######################################################################################################
#Layer (type)                 Output Shape              Param #
#=================================================================
#input_data (InputLayer)      (None, 6, 1)              0
#_________________________________________________________________
#conv1d_1 (Conv1D)            (None, 6, 32)             128
#_________________________________________________________________
#conv1d_2 (Conv1D)            (None, 6, 32)             3104
#_________________________________________________________________
#conv1d_3 (Conv1D)            (None, 6, 32)             3104
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 192)               0
#_________________________________________________________________
#dense_final (Dense)          (None, 6)                 1158
#=================================================================
#Total params: 7,494
#Trainable params: 7,494
#Non-trainable params: 0
#_________________________________________________________________
#-----------------
#SMAPE:
#12.149058606678174









