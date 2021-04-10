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

class Action():
    def __init__(self, hold, stock, last_day):
        self.hold = hold                # holding prices
        self.stock = stock              # Own stock or not
        self.last = last_day            # Save yesterdays prices
        self.action = []                # Save operation log

    def act(self, prices):
        for i in range(prices.shape[0] - 1):
            self.last = prices[i]
            if i == 0:                  # Skip the first day
                self.action.append(0)
            else:
                if self.stock == 1:
                    if prices[i + 1] >= self.hold + 0:        #sell condiction +0
                        self.hold = 0
                        self.action.append(-1)
                        self.stock -= 1
                    else:
                        self.action.append(0)

                elif self.stock == 0:
                    if prices[i + 1] < self.last:
                        self.hold = prices[i + 1]
                        self.action.append(1)
                        self.stock += 1
                    elif prices[i + 1] > self.last:
                        self.hold = prices[i + 1]
                        self.action.append(-1)
                        self.stock -= 1
                    else:
                        self.action.append(0)

                elif self.stock == -1:
                    if prices[i + 1] <= self.hold - 0:        #buy condiction -0
                        self.hold = 0                     
                        self.action.append(1)
                        self.stock += 1
                    else:
                        self.action.append(0)

        return self.action

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

class preprocessing():    
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def data_load(self):
        self.train = pd.read_csv(self.train,  header = None)
        self.test = pd.read_csv(self.test, header = None)
        self.train = pd.concat([self.train,self.test], axis = 0)

    def select_feature(self, feature_num, feature):
        if feature is True:
            self.train_set = self.train.iloc[:,feature_num]  
        else:
            return self.test.iloc[:,0].values     

    def data_scaler(self, scaler, prices, inverse):
        if inverse is False:
            self.train_set = self.train_set.values.reshape(-1,1)
            self.training_set_scaled = scaler.fit_transform(self.train_set)
        else:
            return scaler.inverse_transform(prices)

    def build_train_data(self, scope):
        self.training_data = self.training_set_scaled[0:-20]
        X_train = []
        y_train = []

        for i in range(scope, len(self.training_data), 1):
            X_train.append(self.training_data[i-scope: i, 0])
            y_train.append(self.training_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        return X_train, y_train
    
    def build_test_data(self, scope):
        self.testing_data = self.training_set_scaled[-35:]
        X_test = []
        y_test = []

        for i in range(scope, len(self.testing_data)):
            X_test.append(self.testing_data[i-scope: i, 0])
            y_test.append(self.testing_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        return X_test, y_test

if __name__ == '__main__':
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


    feature_num = 3                                 #We choose the fourth feature(close value) as training data
    sc = MinMaxScaler(feature_range = (0, 1))       #Data scaler
    training_scope = 15                             #We use path 15 days price to predict the next day

    ###Load Data
    load_data = preprocessing(args.training, args.testing)
    load_data.data_load()
    load_data.select_feature(feature_num, feature = True)
    load_data.data_scaler(sc, 0, inverse = False)
    X_train, y_train = load_data.build_train_data(training_scope)
    X_test, y_test = load_data.build_test_data(training_scope)

    ### predicted stock price
    predicted_stock_price = regressor(X_train,y_train).predict(X_test, batch_size = 1)
    predicted_stock_price = load_data.data_scaler(sc, predicted_stock_price, inverse = True)

    Ac = Action(hold = 0, stock = 0, last_day = 0)
    action = Ac.act(predicted_stock_price)

    testing_data = load_data.select_feature(feature_num, feature = False)
    plt.plot(predicted_stock_price, color = 'red', label = 'predict')
    plt.plot(testing_data, color = 'black', label = 'ans')
    plt.legend()
    plt.show()

    print(predicted_stock_price)
    with open(args.output, 'w') as output_file:
        for i in range(len(action)):
            output_file.writelines(str(action[i])+"\n")
