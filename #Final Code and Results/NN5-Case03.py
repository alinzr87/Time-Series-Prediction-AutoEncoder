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
data= pd.read_csv("Data/NN5_interpolated.csv",header=None)
# Expert Knowledge
predictions = pd.read_csv("Data/NN5_theta_25_horg.csv",index_col=0,skiprows = [1])
print(data.shape)
print(predictions.shape)
########################################################################################
########################################################################################
# Functions to Make Inputs

def make_input(data,window_size,horizon=1):
    length=data.shape[0]
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
    data = np.append(data,np.zeros([horizon-extra]))
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
# Auto-Encoder for Expert Knowledge

#$%^^theta_length = predictions.shape[0]
#$%^^horizon=56
#$%^^num_test=56
#$%^^window_size=56
#$%^^theta_all=np.zeros(0)
#$%^^
#$%^^for i in range(theta_length):
#$%^^    current_pred= np.asarray(predictions.iloc[i].dropna().values,dtype=float)
#$%^^    pred_row=current_pred[:-num_test]
#$%^^    pred_auto=make_k_input(pred_row,window_size,horizon)
#$%^^    pred_auto=pred_auto.reshape(pred_auto.shape[0],pred_auto.shape[1])
#$%^^    scaler = MinMaxScaler()
#$%^^    
#$%^^    for j in range(pred_auto.shape[0]):
#$%^^        scaler.fit(pred_auto[j].reshape(pred_auto[j].shape[0],1))
#$%^^        pred_auto[j]= scaler.transform(pred_auto[j].reshape(pred_auto[j].shape[0],1)).reshape(pred_auto[j].shape[0])
#$%^^    
#$%^^    if(i==0):
#$%^^        theta_all=pred_auto
#$%^^    else:
#$%^^        theta_all=np.concatenate((theta_all,pred_auto),axis=0)        
#$%^^normalized_theta_input=theta_all
#$%^^
#$%^^norm_reshaped_input=normalized_theta_input.reshape(normalized_theta_input.shape[0],56)
#$%^^ncol = normalized_theta_input.shape[1] #56
#$%^^encoding_dim = 32
#$%^^input_dim = Input(shape = (ncol, ))
#$%^^# Encoder Layers
#$%^^encoded1 = Dense(48, activation = 'relu')(input_dim)
#$%^^encoded2 = Dense(40, activation = 'relu')(encoded1)
#$%^^encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)
#$%^^# Decoder Layers
#$%^^decoded1 = Dense(40, activation = 'relu')(encoded3)
#$%^^decoded2 = Dense(48, activation = 'sigmoid')(decoded1)
#$%^^decoded3 = Dense(ncol, activation = 'sigmoid')(decoded2)
#$%^^# Combine Encoder and Deocder layers
#$%^^autoencoder = Model(inputs = input_dim, outputs = decoded3)
#$%^^autoencoder.compile(optimizer = 'adam', loss = 'mse')
#$%^^autoencoder.fit(norm_reshaped_input, norm_reshaped_input, epochs = 200, batch_size = 16, shuffle = False,verbose=2)
#$%^^encoder = Model(inputs = input_dim, outputs = encoded3)
#$%^^encoded_input = Input(shape = (encoding_dim, ))
#$%^^auto_out = np.array(encoder.predict(norm_reshaped_input))
#$%^^
#$%^^new_theta_input=auto_out.reshape(auto_out.shape[0],auto_out.shape[1],1)
#$%^^autoencoder.save('NN5_NetworkWeights/NN5_autoencoder_model.h5')
#$%^^encoder.save('NN5_NetworkWeights/NN5_encoder_model.h5')


########################################################################################
########################################################################################
#Main Code

data_length = data.shape[0]
horizon=56
window_size=56
num_test=56
for i in range(data_length):
    current_row= np.asarray(data.iloc[i].dropna().values,dtype=float)
    rr = current_row.size
    rr = int(np.floor(rr*.25))
    series_d=current_row[rr:]
    series_data = series_d[:-num_test]
    series_length = series_data.size
    n_val = int(np.round(series_length*.2))  
    train = series_data[:-n_val]
    val = series_data[-(n_val+window_size):]
    train_sequence = make_input(train, window_size,horizon)
    val_sequence = make_input(val,window_size,horizon)
    
    temp_train_x=train_sequence[0]
    temp_train_x=temp_train_x.reshape(temp_train_x.shape[0],temp_train_x.shape[1])
    temp_train_y=train_sequence[1]
    
    temp_val_x=val_sequence[0]
    temp_val_x=temp_val_x.reshape(temp_val_x.shape[0],temp_val_x.shape[1])
    temp_val_y=val_sequence[1]
    if(i==0):
        data_train_x=temp_train_x
        data_train_y=temp_train_y
        data_val_x=temp_val_x
        data_val_y=temp_val_y
    else:
        data_train_x=np.concatenate((data_train_x,temp_train_x),axis=0)
        data_train_y=np.concatenate((data_train_y,temp_train_y),axis=0)
        data_val_x=np.concatenate((data_val_x,temp_val_x),axis=0)
        data_val_y=np.concatenate((data_val_y,temp_val_y),axis=0)
##################################
##################################
    current_pred= np.asarray(predictions.iloc[i].dropna().values,dtype=float)
    series_p=current_pred
    series_pred=series_p[:-num_test]    
    train_p = series_pred[:-n_val]                                        
    val_p = series_pred[-(n_val+window_size):]
    train_pred = make_k_input(train_p,window_size,horizon)
    val_pred = make_k_input(val_p,window_size,horizon)
    
    temp_train_p_x=train_pred
    temp_train_p_x=temp_train_p_x.reshape(temp_train_p_x.shape[0],temp_train_p_x.shape[1])
    
    temp_val_p_x=val_pred
    temp_val_p_x=temp_val_p_x.reshape(temp_val_p_x.shape[0],temp_val_p_x.shape[1])
    
    scaler = MinMaxScaler()
    for j in range(temp_train_p_x.shape[0]):
        scaler.fit(temp_train_p_x[j].reshape(temp_train_p_x[j].shape[0],1))
        temp_train_p_x[j]= scaler.transform(temp_train_p_x[j].reshape(temp_train_p_x[j].shape[0],1)).reshape(temp_train_p_x[j].shape[0])
    for j in range(temp_val_p_x.shape[0]):
        scaler.fit(temp_val_p_x[j].reshape(temp_val_p_x[j].shape[0],1))
        temp_val_p_x[j]= scaler.transform(temp_val_p_x[j].reshape(temp_val_p_x[j].shape[0],1)).reshape(temp_val_p_x[j].shape[0])
    
    ncol = 56
    encoding_dim = 32
    input_dim = Input(shape = (ncol, ))
    warnings.filterwarnings('ignore')

    input_dim = Input(shape = (ncol, ))
    encoded_input = Input(shape = (encoding_dim, ))
    encoder=load_model('NN5_NetworkWeights/NN5_encoder_model.h5')
    
    temp_train_p_x=np.array(encoder.predict(temp_train_p_x))
    temp_val_p_x=np.array(encoder.predict(temp_val_p_x))
    
    if(i==0):
        data_train_p_x=temp_train_p_x
        data_val_p_x=temp_val_p_x
    else:
        data_train_p_x=np.concatenate((data_train_p_x,temp_train_p_x),axis=0)
        data_val_p_x=np.concatenate((data_val_p_x,temp_val_p_x),axis=0)
##################################
##################################
data_train_x=data_train_x.reshape(data_train_x.shape[0],data_train_x.shape[1],1)
data_train_y=data_train_y.reshape(data_train_y.shape[0],data_train_y.shape[1],1)
data_val_x=data_val_x.reshape(data_val_x.shape[0],data_val_x.shape[1],1)
data_val_y=data_val_y.reshape(data_val_y.shape[0],data_val_y.shape[1],1)

scaler = MinMaxScaler()
for j in range(data_train_x.shape[0]):
    scaler.fit(data_train_x[j])
    data_train_x[j]= scaler.transform(data_train_x[j])
    data_train_y[j]= scaler.transform(data_train_y[j])
for j in range(data_val_x.shape[0]):
    scaler.fit(data_val_x[j])
    data_val_x[j]= scaler.transform(data_val_x[j])
    data_val_y[j]= scaler.transform(data_val_y[j])
data_train_y=data_train_y.reshape(data_train_y.shape[0],data_train_y.shape[1])
data_val_y=data_val_y.reshape(data_val_y.shape[0],data_val_y.shape[1])
#########################################################################################
#######################################################################################
warnings.filterwarnings('ignore')
K.clear_session()
input_data= Input(batch_shape=(None,window_size,1),name='input_data')
input_pred=Input(batch_shape=(None,32),name='input_pred')

encoded_0=Reshape((1,32))(input_pred)
branch_0 = Conv1D(32,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(input_data)
branch_0=concatenate([branch_0,encoded_0],axis=1)
branch_1 = Conv1D(64,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(branch_0)
branch_2 = Conv1D(64,3, strides=1, padding='same',activation='relu',kernel_initializer=glorot_uniform(1))(branch_1)

branch_5=Flatten()(branch_2)

net= Dense(horizon,name='dense_final',activity_regularizer=regularizers.l2(0.00001))(branch_5)

model=Model(inputs=[input_data,input_pred],outputs=net)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00001))
model.fit({'input_data':data_train_x,'input_pred':data_train_p_x},data_train_y,validation_data=[[data_val_x,data_val_p_x],data_val_y],batch_size=16,shuffle=True, epochs=100,verbose=0)
model.save('NN5_NetworkWeights/NN5_Model_03.h5')

warnings.filterwarnings('ignore')
with tqdm(total=data_length) as pbar:
    final_predictions = np.zeros([data_length,56])
    final_test = np.zeros([data_length,56])
    final_smape=np.zeros(data_length)
    for y in range(data_length):                
        current_row =np.asarray(data.loc[y].dropna().values,dtype=float)
        rr = current_row.size
        rr = int(np.floor(rr*.25))
        series_d=current_row[rr:] 
        series_data = series_d[:-num_test]
        series_length = series_data.size
        current_test=series_d[-num_test:]
        n_val = int(np.round(series_length*.2))
               
        test = series_d[-(num_test+window_size):]
        test_sequence = nonov_make_input(test,window_size,horizon)
        test_sequence_norm = deepcopy(test_sequence)
        
        scaler = MinMaxScaler()
        
        for j in range(test_sequence[0].shape[0]):
            scaler.fit(test_sequence[0][j])
            test_sequence_norm[0][j]= scaler.transform(test_sequence[0][j])
            test_sequence_norm[1][j]= scaler.transform(test_sequence[1][j].reshape(test_sequence[1][j].shape[0],1)).reshape(test_sequence[1][j].shape[0])
            
        x_test = test_sequence_norm[0]
        y_test = test_sequence_norm[1]             

        test_input = x_test
                        
        
        current_pred= np.asarray(predictions.iloc[y].dropna().values,dtype=float)
        series_p=current_pred
        series_pred=series_p[:-num_test]
        
        test_p = series_p[-(num_test+window_size):]
        test_pred = nonov_make_k_input(test_p,window_size,horizon)        
                
        test_pred_norm=np.zeros(test_pred.shape)
    
        scaler_pred = MinMaxScaler()
        for j in range(test_pred.shape[0]):            
            scaler_pred.fit(test_pred[j])
            test_pred_norm[j]= scaler_pred.transform(test_pred[j])
        
        test_pred_norm=test_pred_norm.reshape(test_pred_norm.shape[0],test_pred_norm.shape[1])
##########################################################################################################################
        ncol = 56
        encoding_dim = 32
        input_dim = Input(shape = (ncol, ))

        input_dim = Input(shape = (ncol, ))
        encoded_input = Input(shape = (encoding_dim, ))
        encoder=load_model('NN5_NetworkWeights/NN5_encoder_model.h5')
        test_pred_auto=np.array(encoder.predict(test_pred_norm))        
        test_pred_auto=pd.DataFrame(test_pred_auto)    

        
        pred=model.predict({'input_data':test_input, 'input_pred':test_pred_auto})
        pred=scaler.inverse_transform(pred)

        final_predictions[y,:num_test] = pred.reshape(num_test)
        final_test[y,:num_test]=test[-num_test:]

        AAA=smape(pred.reshape(num_test),current_test)       
        final_smape[y]=AAA
        pbar.update(1)

np.savetxt('Output/NN5_Prediction_03.csv',final_predictions, fmt='%1.3f',delimiter=',')
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
#Layer (type)                    Output Shape         Param #     Connected to
#==================================================================================================
#input_data (InputLayer)         (None, 56, 1)        0
#__________________________________________________________________________________________________
#input_pred (InputLayer)         (None, 32)           0
#__________________________________________________________________________________________________
#conv1d_1 (Conv1D)               (None, 56, 32)       128         input_data[0][0]
#__________________________________________________________________________________________________
#reshape_1 (Reshape)             (None, 1, 32)        0           input_pred[0][0]
#__________________________________________________________________________________________________
#concatenate_1 (Concatenate)     (None, 57, 32)       0           conv1d_1[0][0]
#                                                                 reshape_1[0][0]
#__________________________________________________________________________________________________
#conv1d_2 (Conv1D)               (None, 57, 64)       6208        concatenate_1[0][0]
#__________________________________________________________________________________________________
#conv1d_3 (Conv1D)               (None, 57, 64)       12352       conv1d_2[0][0]
#__________________________________________________________________________________________________
#flatten_1 (Flatten)             (None, 3648)         0           conv1d_3[0][0]
#__________________________________________________________________________________________________
#dense_final (Dense)             (None, 56)           204344      flatten_1[0][0]
#==================================================================================================
#Total params: 223,032
#Trainable params: 223,032
#Non-trainable params: 0
#__________________________________________________________________________________________________
#-----------------
#SMAPE:
#20.75328220918917
