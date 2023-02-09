import pandas as pd
from sklearn.model_selection import train_test_split
import para_cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold



if __name__ == '__main__':
    ml_train = pd.read_csv('ML-CUP21-TR - Copy.csv', sep=',', header=None)
    ml_train.drop(0, axis = 1, inplace=True)
    rename_dict = {}
    for i in range(1,11): rename_dict[i] = f"attr_{i}"
    rename_dict.update({11:'target_1', 12:'target_2'})
    ml_train.rename( columns=rename_dict, inplace =True)
    X = ml_train.iloc[:,:10]
    y = ml_train.loc[:,['target_1','target_2']]
    X, X_test_final, y, y_test_final = train_test_split(X, y, test_size=0.1) 
    def build_model_sgd(l_rate, mom=0, decay=0, nesterov = False, weights_init = 'glorot_uniform'):
        model = Sequential()
        model.add(Dense(10 ,input_dim=X.shape[1], activation = 'relu', kernel_initializer=weights_init))
        model.add(Dense(2))
        opt = SGD(learning_rate=l_rate, momentum=mom, decay= decay, nesterov=nesterov)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['MAE'])
        return model
    #model = build_model_sgd(0.003, 0.9, 0.001/100, True, 'he_uniform')
    kfold = KFold(n_splits=10, shuffle=True)
    res = para_cv.launch(X = X,
                y = y,
                kfold = kfold,
                epochs = 10,
                batch_size = 32)
    print(res)

