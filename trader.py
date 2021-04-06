import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
from sklearn.preprocessing import MinMaxScaler 


def regressor(X_train,y_train):
    keras.backend.clear_session()
    regressor = Sequential()
    regressor.add(LSTM(units = 100,input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 10, batch_size = 16)

    return regressor

    
if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    # load_data
    testing_data = pd.read_csv(args.testing, header = None)
    training_data = pd.read_csv(args.training,  header = None)

    # select feature
    train_set = training_data.iloc[:,0]
    test_set = testing_data.iloc[:,0]

    # data scaler
    sc = MinMaxScaler(feature_range = (0, 1))
    train_set = train_set.values.reshape(-1,1)
    training_set_scaled = sc.fit_transform(train_set)

    # build train data
    X_train = [] 
    y_train = []
    for i in range(10,len(train_set)):
        X_train.append(training_set_scaled[i-10:i-1, 0]) 
        y_train.append(training_set_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train) 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # build test data
    dataset_total = pd.concat((training_data.iloc[:,0], testing_data.iloc[:,0]), ignore_index = True, axis = 0)
    inputs = dataset_total[len(dataset_total) - len(testing_data) - 10:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs)
    
    X_test = []
    for i in range(10, len(inputs)):
        X_test.append(inputs[i-10:i-1, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # predicted stock price
    predicted_stock_price = regressor(X_train,y_train).predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    


    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
            
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)
