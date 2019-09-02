import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# 数据手动处理 预测数据另成一列 上移一格 删除最后一行 形成temp.csv文件



# 生成训练和测试数据
def generate_model_data(data_path,alpha,days):
    ## days=15 15天为一个时间步，data_path=temp.csv 读取文件路径，alpha:选取训练数据比例0.8即80%
    df = pd.read_csv(data_path)
    train_day = int((len(df['close']) - days + 1))#701-15+1=687
    for property in ['open', 'close', 'high', 'low', 'volume','next_close']:
        df[property] = scaler.fit_transform(np.reshape(np.array(df[property]), (-1, 1)))#转换到0至1之间
    X_data, Y_data = list(), list()
    # 生成时序数据
    for i in range(train_day):#687
        Y_data.append(df['next_close'][i+days-1])#14 15 16......700
        for j in range(days):#15
            for m in ['open', 'close', 'high', 'low', 'volume']:
                X_data.append(df[m][i + j])
    # print(len(df['next_close']))
    # print(Y_data)
    #print(len(Y_data))
    #print(X_data)
    #print(len(X_data))#687*15*5=51525

    #print(X_data)
    X_data = np.reshape(np.array(X_data),(-1,5*days))# 5表示特征数量*天数  #687行 15*5列 二维数组
    #print(X_data)
    train_length = int(len(Y_data)* alpha)  #687*0.8=549.6  ===549
    # A=X_data[:train_length]
    # print(A)
    # print(len(A))
    X_train = np.reshape(np.array(X_data[:train_length]),(len(X_data[:train_length]),days,5))#三维数组：每个二维15行5列
    #print(X_train)
    print(len(X_train))
    X_test = np.reshape(np.array(X_data[train_length:]),(len(X_data[train_length:]),days,5))
    Y_train,Y_test = np.array(Y_data[:train_length]),np.array(Y_data[train_length:])
    #print(len(X_test))#138
   # print(len(Y_train))#549
    #print(len(Y_test))#138
    return X_train,Y_train,X_test,Y_test

#MAPE
def calc_MAPE(real,predict):
    Score_MAPE = 0
    for i in range(len(predict[:, 0])):#所有行第一列
        Score_MAPE += abs((predict[:, 0][i] - real[:, 0][i]) / real[:, 0][i])
    Score_MAPE = Score_MAPE  / len(predict[:, 0])
    return Score_MAPE

def calc_AMAPE(real,predict):
    Score_AMAPE = 0
    Score_MAPE_DIV = sum(real[:, 0]) / len(real[:, 0])#平均值
    for i in range(len(predict[:, 0])):
        Score_AMAPE += abs((predict[:, 0][i] - real[:, 0][i]) / Score_MAPE_DIV)
    Score_AMAPE = Score_AMAPE / len(predict[:, 0])
    return Score_AMAPE

def evaluate(real,predict):
    RMSE = math.sqrt(mean_squared_error(real[:, 0], predict[:, 0]))
    MAE = mean_absolute_error(real[:, 0], predict[:, 0])
    MAPE = calc_MAPE(real, predict)
    AMAPE = calc_AMAPE(real, predict)
    return RMSE,MAE,MAPE,AMAPE

def lstm_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(units=20, input_shape=(X_train.shape[1], X_train.shape[2])))#15*5 units:输出维度
    model.add(Dense(1,input_dim=(days,20),activation='hard_sigmoid'))##########原本没有输入维度
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=200, batch_size=20, verbose=1)#model.fit 进行训练

    trainPredict = model.predict(X_train)  #返回预测值trainPredict
    trainPredict = scaler.inverse_transform(trainPredict)#将标准化后的数据转换为原来数据
    Y_train = scaler.inverse_transform(np.reshape(Y_train, (-1, 1)))

    testPredict = model.predict(X_test)
    testPredict = scaler.inverse_transform(testPredict)
    Y_test = scaler.inverse_transform(np.reshape(Y_test, (-1, 1)))

    return Y_train, trainPredict, Y_test, testPredict
if __name__=='__main__':
    #在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而import到其他脚本中是不会被执行的。
    days = 15
    alpha = 0.8
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, Y_train, X_test, Y_test = generate_model_data('temp.csv',alpha,days)
    train_Y, trainPredict, test_Y, testPredict  = lstm_model(X_train, Y_train, X_test, Y_test)
    plt.plot(list(trainPredict), color='red', label='prediction')#转换为列表
    plt.plot(list(train_Y), color='blue', label='real')
    plt.legend(loc='upper left')#图例位置
    plt.title('train data')
    plt.show()
    plt.plot(list(testPredict), color='red', label='prediction')
    plt.plot(list(test_Y), color='blue', label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.show()
    #
    RMSE,MAE,MAPE,AMAPE = evaluate(test_Y,testPredict)
    print(RMSE,MAE,MAPE,AMAPE)