import tensorflow as tf
import numpy as np
import pandas as pd
import sys

roundT=20 #训练轮次数
learnRateT=0.05#学习率

#命令行参数
argt=sys.argv[1:]
print('argt:%s'%argt)

for v in argt:
    if v.startswith('-round='):
        roundT=int(v[len('-round='):])
    if v.startswith('-learnrate='):
        learnRateT=float(v[len('-learnrate='):])

fileData=pd.read_csv('exchangeData3.csv',dtype=np.float32,header=None,usecols=(1,2,3,4,5,6,7))
wholeData=np.reshape(fileData.as_matrix(),(-1,7))#7列

print('wholeData:%s'%wholeData)#获取整个数据

cellCount=4#三个时刻为一批
unitCount=5 #每个结构元有5个神经元节点

testData=wholeData[504:]
print('testData:%s\n'%testData)   #二维数组

testCount=testData.shape[0]-cellCount
print('testCount:%s\n'%testCount)

x_testData=[testData[i:i+cellCount,0:6]for i in range(testCount)]
y_testData=[testData[i+cellCount,6]for i in range(testCount)]

print('x_testData:%s'%x_testData)
print('y_testData:%s'%y_testData)

rowCount=wholeData.shape[0]-testData.shape[0]#504
print('rowCount:%d\n'%rowCount)

xData=[wholeData[i:i+cellCount,0:6]for i in range(rowCount)]
yTrainData=[wholeData[i+cellCount,6]for i in range(rowCount)]

print('xData:%s'%xData)
print('yTrainData:%s'%yTrainData)

x=tf.placeholder(shape=[cellCount,6],dtype=tf.float32)
y=tf.placeholder(dtype=tf.float32)

cellT=tf.nn.rnn_cell.BasicLSTMCell(unitCount)#定义一个LSTM结构单元，并指定神经元节点数量

initState=cellT.zero_state(1,dtype=tf.float32)

h,finalState=tf.nn.dynamic_rnn(cellT,tf.reshape(x,[1,cellCount,6]),initial_state=initState,dtype=tf.float32)

hr=tf.reshape(h,[cellCount,unitCount])

w2=tf.Variable(tf.random_normal([unitCount,1]),dtype=tf.float32)

b2=tf.Variable(tf.random_normal([1]),dtype=tf.float32)

predicit=tf.reduce_sum(tf.matmul(hr,w2)+b2)#行维度求和

loss=tf.abs((predicit-y)/y)

optimizer=tf.train.RMSPropOptimizer(learnRateT)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(roundT):
    lossSum=0.0
    for j in range(rowCount):
        result = sess.run([train, x, y, predicit , h, finalState, loss], feed_dict={x: xData[j], y: yTrainData[j]})

        lossSum = lossSum + float(result[len(result) - 1])
        if j == (rowCount - 1):
            print('i:%d,x:%s,y:%s,predicit:%s,h:%s,finalState:%s,loss:%s,avgLoss:%10.10f\n' % (
            i, result[1], result[2], result[3], result[4], result[5], result[6], (lossSum / rowCount)))



test_lossSum=0.0
for i in range(testCount):
    result = sess.run([x, predicit,loss], feed_dict={x: x_testData[i],y:y_testData[i]})

    test_lossSum=test_lossSum + float(result[len(result) - 1])
    if i==(testCount-1):
         print('y:%s,MAPE:%10.10f\n'  %(result[1],(test_lossSum /testCount )))
