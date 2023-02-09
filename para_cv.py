from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold

def build_model_sgd(l_rate, mom=0, decay=0, nesterov = False, weights_init = 'glorot_uniform'):
    model = Sequential()
    model.add(Dense(10 ,input_dim=10, activation = 'relu', kernel_initializer=weights_init))
    model.add(Dense(2))
    opt = SGD(learning_rate=l_rate, momentum=mom, decay= decay, nesterov=nesterov)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['MAE'])
    return model

def single_cross_val(train, test, X, y, shared_dic, k):
    model = build_model_sgd(0.003, 0.9, 0.001/1000, True, 'he_uniform')
    x = model.fit(X.iloc[train,:], y.iloc[train,:], validation_data= (X.iloc[test,:], y.iloc[test,:]), epochs=1000, batch_size=32, verbose=0).history
    shared_dic[f'key_{k}'] = x

def launch( X, y, kfold, epochs, batch_size):
    shared_dic = Manager().dict()
    #kfold = KFold(n_splits=n_splits, shuffle=True)
    k = 0
    with ProcessPoolExecutor() as executor:
        for train, test in kfold.split(X, y):
            executor.submit(single_cross_val, train, test, X, y, shared_dic, k)
            k += 1
    return dict(shared_dic)

