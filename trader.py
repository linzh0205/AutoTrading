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


def predict_action(predicted_stock_price):   
    stock = 0
    money = 0
    action = []
    for i in range(predicted_stock_price.shape[0]-1):
        if i == 0:
            continue
        else:
            if stock == 1:
                if predicted_stock_price[i-1]<predicted_stock_price[i]:
                    action.append(-1)
                elif predicted_stock_price[i-1]>predicted_stock_price[i]:
                    action.append(0)

            elif stock == 0:
                if predicted_stock_price[i-1]<predicted_stock_price[i]:
                    action.append(0)
                    money += predicted_stock_price[i]
                elif predicted_stock_price[i-1]>predicted_stock_price[i]:
                    action.append(0)
                    money += predicted_stock_price[i]
                    
            elif stock == -1:
                if predicted_stock_price[i-1]<predicted_stock_price[i]:
                    action.append(0)
                elif predicted_stock_price[i-1]>predicted_stock_price[i]:
                    action.append(1)

        stock += action[-1]

    return action

def regressor(X_train,y_train):
    keras.backend.clear_session()
    regressor = Sequential()

    regressor.add(LSTM(units = 16,
                    batch_input_shape = (1 ,X_train.shape[1], 1),
                    stateful= True
                    ))
    
    regressor.add(Dense(units = 1))

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

    
    testing_data = testing_data.iloc[:,0].values
    action = predict_action(predicted_stock_price)


    plt.plot(predicted_stock_price, color = 'red', label = 'predict')
    plt.plot(testing_data, color = 'black', label = 'ans')
    plt.legend()
    plt.show()


    with open(args.output, 'w') as output_file:
        for i in range(len(action)):
            output_file.writelines(str(action[i])+"\n")