import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras import regularizers


def regressor(X_train,y_train):
    # keras.backend.clear_session()
    # regressor = Sequential()
    # regressor.add(LSTM(units = 50,input_shape = (X_train.shape[1], 1)))
    # regressor.add(Dropout(0.2))
    # regressor.add(Dense(units = 1))
    # regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # regressor.fit(X_train, y_train, epochs = 10, batch_size = 16)

    keras.backend.clear_session()
    regressor = Sequential()

    regressor.add(LSTM(units = 16,
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    batch_input_shape = (1 ,X_train.shape[1], 1),
                    # bias_regularizer=regularizers.l2(1e-4),
                    # activity_regularizer=regularizers.l2(1e-5),
                    # return_sequences = True,
                    stateful= True
                    ))
    

    # regressor.add(LSTM(units = 16,
    #                 # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    #                 # batch_input_shape = (1 ,X_train.shape[1], 1),
    #                 # bias_regularizer=regularizers.l2(1e-4),
    #                 # activity_regularizer=regularizers.l2(1e-5),
    #                 return_sequences = True,
    #                 stateful= True
    #                 ))
    
    # regressor.add(Dropout(0.7))

    # regressor.add(LSTM(units = 16,
    #                 # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    #                 # batch_input_shape = (1 ,X_train.shape[1], 1),
    #                 # bias_regularizer=regularizers.l2(1e-4),
    #                 # activity_regularizer=regularizers.l2(1e-5),
    #                 stateful= True
    #                 ))

    # regressor.add(Dropout(0.7))

    regressor.add(Dense(units = 1,
                        # kernel_regularizer=regularizers.l1_l2(l1= 0, l2=0.01),
                        # bias_regularizer=regularizers.l2(0.01),
                        # activity_regularizer=regularizers.l2(0.01)
                        ))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 10, batch_size = 1)
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
    testing_data = pd.read_csv(args.testing, header = None)         #(250, 4)
    training_data = pd.concat([training_data,testing_data], axis = 0)   #(1508, 4)


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
    train_set = training_data.iloc[:,3]
    # test_set = testing_data.iloc[:,0]

    # data scaler
    sc = MinMaxScaler(feature_range = (0, 1))       #(1508,)
    train_set = train_set.values.reshape(-1,1)      #(1508, 1)
    training_set_scaled = sc.fit_transform(train_set)
    
    training_data = training_set_scaled[:1488]
    # testing_data = training_set_scaled[1488:]

    # build train data
    X_train = []
    y_train = []
    for i in range(15, len(training_data), 1):
        X_train.append(training_data[i-15:i, 0])
        y_train.append(training_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # print(X_train.shape)      #(1498, 10)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #(1498, 10, 1)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))                    #(1498, 1)

    # build test data
    # inputs = training_set_scaled[1478:]     #(260, 2)
    inputs = training_set_scaled[1473:]     #(260, 2)

    X_test = []
    y_test = []
    for i in range(15, len(inputs)):
        X_test.append(inputs[i-15:i, 0])
        y_test.append(inputs[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    # print(X_test.shape)
    
    # predicted stock price
    predicted_stock_price = regressor(X_train,y_train).predict(X_test, batch_size = 1)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    for i in range(predicted_stock_price.shape[0]):
        print(str(i) + ' ' + str(predicted_stock_price[i]))

    testing_data = testing_data.iloc[:,0].values
    plt.plot(predicted_stock_price, color = 'red', label = 'predict')
    plt.plot(testing_data, color = 'black', label = 'ans')
    plt.legend()
    plt.show()

    with open(args.output, 'w') as output_file:
        for row in testing_data:
            
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row)
            output_file.write(action)