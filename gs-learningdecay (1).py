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

ml_train = pd.read_csv('ML-CUP21-TR - Copy.csv', sep=',', header=None)
ml_train.drop(0, axis = 1, inplace=True)
rename_dict = {}
for i in range(1,11): rename_dict[i] = f"attr_{i}"
rename_dict.update({11:'target_1', 12:'target_2'})
ml_train.rename( columns=rename_dict, inplace =True)
X = pd.read_csv('X.csv')
X_test_final = pd.read_csv('X_test_final.csv')
y = pd.read_csv('y.csv')
y_test_final = pd.read_csv('y_test_final.csv')



def build_model_sgd_3hl_for_gridsearch(neurons_l1,neurons_l2,neurons_l3, l_rate, mom=0, decay=0, weights_init = 'he_uniform', input_dim = X.shape[1]):
    model = Sequential()
    model.add(Dense(neurons_l1, input_dim=input_dim, activation = 'relu', kernel_initializer=weights_init))
    model.add(Dense(neurons_l2, activation='relu'))
    model.add(Dense(neurons_l3, activation='relu'))
    model.add(Dense(2))
    opt = SGD(learning_rate=l_rate, momentum=mom, decay=decay , nesterov=True, clipnorm=10)
    model.compile(loss=MSE, optimizer=opt, metrics=[MEE])
    return model

def parallel_cv_gridsearch_3hl(train,test,neurons_l1, neurons_l2,neurons_l3, l_rate, mom, decay, weights_init, batch_size):
    model = build_model_sgd_3hl_for_gridsearch(neurons_l1,neurons_l2,neurons_l3, l_rate, mom, decay, weights_init)
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, min_delta = 0.001, verbose = 0, restore_best_weights=True)
    model.fit(X.iloc[train,:], y.iloc[train,:], validation_data= (X.iloc[test,:], y.iloc[test,:]), epochs=700, batch_size=batch_size, verbose=0, callbacks=[es]).history
    scores = model.evaluate(X.iloc[test,:], y.iloc[test], verbose=0)
    return scores

# K-fold Cross Validation model evaluation
from sklearn.model_selection import ParameterGrid
num_folds = 7
kfold = KFold(n_splits=num_folds, shuffle=True)

param_dict = {
    'mom' : [0.9],
    'l_rate' : np.arange(0.001,0.009, 0.0005),
    'decay' : np.arange(1e-3, 9e-4,0.002),
    }

grid = list(ParameterGrid(param_dict))
df_params = pd.DataFrame(ParameterGrid(param_dict))
print(f'{num_folds}-CV Gridsearch\nTesting {len(grid)} parameter combinations')
for i in df_params.index:
    futures = []
    for train,test in kfold.split(X,y):
        future = client.submit(parallel_cv_gridsearch_3hl,
                                 train=train,
                                 test=test, 
                                 neurons_l1=300,
                                 neurons_l2=300,
                                 neurons_l3=300,
                                 l_rate= df_params.loc[i,'l_rate'],
                                 mom = df_params.loc[i,'mom'],
                                 decay = df_params.loc[i,'decay'],
                                 weights_init = 'glorot_normal',
                                 batch_size = 64)
        futures.append(future)
    scores = client.gather(futures)
    mean_loss=np.mean([scores[_] for _ in range(num_folds)])
    mean_mee=np.mean([scores[_][1] for _ in range(num_folds)])
    df_params.loc[i,'MSE'] = mean_loss
    df_params.loc[i,'MEE'] = mean_mee
    print(dict(df_params.loc[i,]))

df_params.to_csv('gridsearch_resultstest.csv')
print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')
print(df_params.loc[df_params.MEE==df_params.MEE.min(),])
