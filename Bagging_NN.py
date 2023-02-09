
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.callbacks import EarlyStopping

from dask.distributed import Client
client = Client("tcp://10.138.0.2:8786")

from tensorflow.keras import backend
def MEE(y_true, y_pred):
    return backend.sqrt(backend.sum(backend.square(y_true - y_pred), axis=-1, keepdims=True))
def MSE(y_true, y_pred):
    return backend.sum(backend.square(y_true - y_pred), axis=-1, keepdims=True)
def RMSE(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.sum(backend.square(y_true - y_pred), axis=-1, keepdims=True)))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["KMP_SETTINGS"] = "false"
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

ml_train.rename( columns=rename_dict, inplace =True)
X = pd.read_csv('X.csv')
X_test_final = pd.read_csv('X_test_final.csv')
y = pd.read_csv('y.csv')
y_test_final = pd.read_csv('y_test_final.csv')


def build_model_sgd_3hl(neurons_l1,neurons_l2,neurons_l3, l_rate, mom=0, decay=0, nesterov = True, weights_init = 'glorot_normal',loss=MSE,clipnorm=None):
    model = Sequential()
    model.add(Dense(neurons_l1, input_dim=X_train.shape[1], activation = 'relu', kernel_initializer=weights_init))
    model.add(Dense(neurons_l2, activation='relu'))
    model.add(Dense(neurons_l3, activation='relu'))
    model.add(Dense(2))
    opt = SGD(learning_rate=l_rate, momentum=mom, decay= decay, nesterov=nesterov, clipnorm=clipnorm)
    model.compile(loss=loss, optimizer=opt, metrics=[MEE])
    return model


from sklearn.utils import resample

def list_sum(y_pred_bag, n_iter):
    summed = 0
    for i in range(n_iter):
        summed = summed + y_pred_bag[i]
    y_pred_final = summed/n_iter
    return y_pred_final

def parallel_bag(X_tr, y_tr,X_val):
    model = build_model_sgd_3hl(neurons_l1=40,neurons_l2=40,neurons_l3=40, l_rate=0.0012, mom=0.9, decay=0, nesterov = False, weights_init = 'glorot_normal')
    x = model.fit(X_tr, y_tr, epochs = 3000, batch_size = 256,verbose=0).history
    y_pred_rep = model.predict(X_val)
    return y_pred_rep
n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True)
n_iter = 24
MEE_splits=[]
MSE_splits=[]
for train, test in kfold.split(X,y):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train,:]
    X_val = X.iloc[test,:]
    y_val = y.iloc[test,:]
    size = len(X_train)
    futures = []
    for rep in range(n_iter):
        ix = [i for i in range(size)]
        train_ix = resample(ix, replace=True, n_samples=size)
        test_ix = [x for x in ix if x not in train_ix]
        X_tr = X_train.iloc[train_ix,:]
        y_tr = y_train.iloc[train_ix,:]
        X_ts = X_train.iloc[test_ix,:]
        y_ts = y_train.iloc[test_ix,:]
        future = client.submit(parallel_bag, X_tr, y_tr, X_val)
        futures.append(future)
    y_pred_bag = client.gather(futures)
    y_pred_final = list_sum(y_pred_bag,n_iter)
    MEE_splits.append(np.mean(MEE(y_pred_final,y_val)))
    MSE_splits.append(np.mean(MSE(y_pred_final,y_val)))                  
mean_MEE_1 = np.mean(MEE_splits)
mean_MSE_1 = np.mean(MSE_splits)
#y_pred_final = [np.mean(x) for x in zip(*y_pred_bag for _ in range(100))]
    
MEE_splits_2=[]
MSE_splits_2 =[]                      
n_iter = 48
for train, test in kfold.split(X,y):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train,:]
    X_val = X.iloc[test,:]
    y_val = y.iloc[test,:]
    size = len(X_train)
    futures = []
    for rep in range(n_iter):
        ix = [i for i in range(size)]
        train_ix = resample(ix, replace=True, n_samples=size)
        test_ix = [x for x in ix if x not in train_ix]
        X_tr = X_train.iloc[train_ix,:]
        y_tr = y_train.iloc[train_ix,:]
        X_ts = X_train.iloc[test_ix,:]
        y_ts = y_train.iloc[test_ix,:]
        future = client.submit(parallel_bag, X_tr, y_tr, X_val)
        futures.append(future)
    y_pred_bag = client.gather(futures)
    y_pred_final = list_sum(y_pred_bag,n_iter)
    MEE_splits_2.append(np.mean(MEE(y_pred_final,y_val)))
    MSE_splits_2.append(np.mean(MSE(y_pred_final,y_val)))                  
mean_MEE_2 = np.mean(MEE_splits_2)
mean_MSE_2 = np.mean(MSE_splits_2)
                      

MEE_splits_3=[]
MSE_splits_3 =[]                        
n_iter = 96
for train, test in kfold.split(X,y):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train,:]
    X_val = X.iloc[test,:]
    y_val = y.iloc[test,:]
    size = len(X_train)
    futures = []
    for rep in range(n_iter):
        ix = [i for i in range(size)]
        train_ix = resample(ix, replace=True, n_samples=size)
        test_ix = [x for x in ix if x not in train_ix]
        X_tr = X_train.iloc[train_ix,:]
        y_tr = y_train.iloc[train_ix,:]
        X_ts = X_train.iloc[test_ix,:]
        y_ts = y_train.iloc[test_ix,:]
        future = client.submit(parallel_bag, X_tr, y_tr, X_val)
        futures.append(future)
    y_pred_bag = client.gather(futures)
    y_pred_final = list_sum(y_pred_bag,n_iter)
    MEE_splits_3.append(np.mean(MEE(y_pred_final,y_val)))
    MSE_splits_3.append(np.mean(MSE(y_pred_final,y_val)))                  
mean_MEE_3 = np.mean(MEE_splits_3)
mean_MSE_3 = np.mean(MSE_splits_3)
                        
MEE_splits_4=[]
MSE_splits_4 =[]                        
n_iter = 152
for train, test in kfold.split(X,y):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train,:]
    X_val = X.iloc[test,:]
    y_val = y.iloc[test,:]
    size = len(X_train)
    futures = []
    for rep in range(n_iter):
        ix = [i for i in range(size)]
        train_ix = resample(ix, replace=True, n_samples=size)
        test_ix = [x for x in ix if x not in train_ix]
        X_tr = X_train.iloc[train_ix,:]
        y_tr = y_train.iloc[train_ix,:]
        X_ts = X_train.iloc[test_ix,:]
        y_ts = y_train.iloc[test_ix,:]
        future = client.submit(parallel_bag, X_tr, y_tr, X_val)
        futures.append(future)
    y_pred_bag = client.gather(futures)
    y_pred_final = list_sum(y_pred_bag,n_iter)
    MEE_splits_4.append(np.mean(MEE(y_pred_final,y_val)))
    MSE_splits_4.append(np.mean(MSE(y_pred_final,y_val)))                  
mean_MEE_4 = np.mean(MEE_splits_4)
mean_MSE_4 = np.mean(MSE_splits_4)
                        
print('mean_MEE 1, 24 iter: ', mean_MEE_1)
print('mean_MSE 1, 24 iter: ', mean_MSE_1)
print('mean_MEE 2, 48 iter: ', mean_MEE_2)
print('mean_MSE 2, 48 iter: ', mean_MSE_2)
print('mean_MEE 3, 96 iter: ', mean_MEE_3)
print('mean_MSE 3, 96 iter: ', mean_MSE_3)
print('mean_MEE 4, 152 iter: ', mean_MEE_4)
print('mean_MSE 4, 152 iter: ', mean_MSE_4)
