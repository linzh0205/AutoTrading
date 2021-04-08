import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler 


def regressor(X_train,y_train):
    # keras.backend.clear_session()
    # regressor = Sequential()
    # regressor.add(LSTM(units = 64,return_sequences = True  ,input_shape = (X_train.shape[1], 2)))
    # regressor.add(Dropout(0.5))
    # regressor.add(LSTM(units = 64,return_sequences = True))
    # regressor.add(Dropout(0.5))
    # regressor.add(LSTM(units = 32) )
    # regressor.add(Dropout(0.5))
    # regressor.add(Dense(units = 1))
    # regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # regressor.fit(X_train, y_train, epochs = 80, batch_size = 32)
    keras.backend.clear_session()
    regressor = Sequential()
    regressor.add(LSTM(units = 16,kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01),
                    input_shape = (X_train.shape[1], 2)))
    
    # batch_input_shape=(batch_size, timesteps, data_dim)
    # regressor.add(Dense(units = 16, batch_input_shape=(batch_size, timesteps, data_dim) , stateful = True))
    # regressor.add(Dense(units = 1 , kernel_regularizers = regularizers.l1_l2(l1 = 0, l2 = 0.01) ,
    #                     bias_regularizer = regularizers.l2(0.01), activity_regularizer= regularizers.l2(0.01)))
    regressor.add(Dense(units = 1,kernel_regularizer=regularizers.l1_l2(l1= 0, l2=0.01),
                        bias_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01)))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 300, batch_size = 8)

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
    training_data = pd.read_csv(args.training,  header = None)      #(1258, 4)
    testing_data = pd.read_csv(args.testing, header = None)         #(20, 4)
    training_data = pd.concat([training_data, testing_data], ignore_index = True, axis = 0)   #(1508, 4)


    # show_open = testing_data.iloc[:, 0]
    # show_high = testing_data.iloc[:, 1]
    # show_low = testing_data.iloc[:, 2]
    # show_close = testing_data.iloc[:, 3]
    # plt.plot(show_open, color = 'red', label = 'show_open')
    # # plt.plot(show_high, color = 'blue', label = 'show_high')
    # # plt.plot(show_low, color = 'black', label = 'show_low')
    # plt.plot(show_close, color = 'black', label = 'show_close')
    # plt.legend()
    # plt.show()


    # select feature
    train_set_open = training_data.iloc[:,0]
    train_set_close = training_data.iloc[:,3]
    training_data = pd.concat([train_set_open,train_set_close], axis = 1)       # (1508, 2)
    # print(training_data)
    # test_set = testing_data.iloc[:,0]

    # data scaler
    sc = MinMaxScaler(feature_range = (0, 1))               #(1508,)
    training_data = training_data.values.reshape(-1,2)      #(1508, 1)
    training_set_scaled = sc.fit_transform(training_data)   #(1508, 2)
    # print(training_set_scaled)
    # print(training_set_scaled.shape)

    training_data = training_set_scaled[:1488]
    # testing_data = training_set_scaled[1488:]

    # build train data
    X_train = []
    y_train = []
    for i in range(2, len(training_data), 1):
        X_train.append(training_data[i-2:i, :])        #(1248, 10, 2)
        y_train.append(training_data[i, 0])             #(1248, 1)
    X_train, y_train = np.array(X_train), np.array(y_train)
    # print(X_train.shape)      #(1498, 10)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))  #(1248, 10, 2)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))                    #(1248, 1)
    
    # build test data
    # inputs = training_set_scaled[1478:]     #(260, 2)
    inputs = training_set_scaled[1486:]     #(260, 2)

    X_test = []
    y_test = []
    for i in range(2, len(inputs)):
        X_test.append(inputs[i-2:i, :])
        y_test.append(inputs[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))      #(250, 10, 2)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))                       #(250, 1)
    # print(X_test.shape)
    # print(y_test.shape)
    
    # predicted stock price
    predicted_stock_price = regressor(X_train,y_train).predict(X_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))  #(1248, 10, 2)
    predicted_stock_price = np.reshape(predicted_stock_price, (-1, 2))
    print(predicted_stock_price.shape)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = np.reshape(predicted_stock_price, (-1, 1))
    for i in range(predicted_stock_price.shape[0]):
        print(str(i + 1) + ' ' + str(predicted_stock_price[i]))

    
    # testing_data = np.array(testing_data.il)
    testing_data = testing_data.iloc[:,0].values
    plt.plot(predicted_stock_price, color = 'red', label = 'predict')
    plt.plot(testing_data, color = 'black', label = 'ans')
    plt.legend()
    plt.show()

    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
            
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)