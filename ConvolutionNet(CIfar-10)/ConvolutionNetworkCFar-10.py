import tensorflow as tf
import loadData as ld
import datetime
import math

"""
This Code is the ConvolutionNeural Network that works on the CIfar-10
data set which has 10 classes.To check which classes are those you can
call load_class_names()in loadData file to see the classes names.This 
is just quick and dirty implementation of network without any pooling or
any batch normalization or data agumentation techniques.

*************************OUTPUT OF LAYER***************************
The output of any layer is computed by
d=(F-W)/S+1
d:OUTPUT DIMENSION
F:THE INPUT DIMENSION
W:FILTER/WEIGHT DIMENSION
S:STRIDE (By how much frames we move forward)
"""

def ConvolutionlNetwork():

    ##hyperParameters
    epochs = 12000
    batchSize = 100
    logsPathTrain = "./logs/train"
    logsPathTest = "./logs/test"
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0

    with tf.name_scope('input'):
        X=tf.placeholder(tf.float32,shape=[None,32,32,3])
        Y=tf.placeholder(tf.float32,shape=[None,10])
        # variable learning rate
        lr = tf.placeholder(tf.float32)
        # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
        pkeep = tf.placeholder(tf.float32)

    with tf.name_scope('weights_and_biases'):
        W1=tf.Variable(tf.truncated_normal([4,4,3,4],stddev=0.1))
        b1=tf.Variable(tf.constant(0.1,tf.float32,[4]))

        W2 = tf.Variable(tf.truncated_normal([4, 4, 4, 8], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, tf.float32, [8]))

        W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, tf.float32, [12]))

        W4 = tf.Variable(tf.truncated_normal([4, 4, 12, 16], stddev=0.1))
        b4 = tf.Variable(tf.constant(0.1, tf.float32, [16]))

        W5 = tf.Variable(tf.truncated_normal([9 *9* 16, 200], stddev=0.1))
        b5 = tf.Variable(tf.constant(0.1, tf.float32, [200]))

        W6 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))  ##will
        b6 = tf.Variable(tf.constant(0.1, tf.float32, [10]))


        ##Architectire Design

    with tf.name_scope('layers'):
        stride = 1  # output 29*29
        hx1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)
        stride = 1  # output 26*26
        hx2 = tf.nn.relu(tf.nn.conv2d(hx1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)
        stride = 2  # output 12*12
        hx3 = tf.nn.relu(tf.nn.conv2d(hx2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)
        stride = 1  # output 9*9
        hx4 = tf.nn.relu(tf.nn.conv2d(hx3, W4, strides=[1, stride, stride, 1], padding='SAME') + b4)

        reshape = tf.reshape(hx4, [-1, 9 * 9 * 16])

        hx4 = tf.nn.relu(tf.matmul(reshape, W5) + b5)
        hx4d = tf.nn.dropout(hx4, pkeep)
        ylogits = tf.matmul(hx4d, W6) + b6
        hx = tf.nn.softmax(ylogits)

    with tf.name_scope("costFunction"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=Y)
        cross_entropy = tf.reduce_mean(cross_entropy) * 100

    with tf.name_scope("accuracy"):
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(hx, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    learning_rate =max_learning_rate
    testX,testY=ld.gettest()
    testData={X:testX,Y:testY,pkeep:1.0}
    print("the Training Starts")
    for epoch in range(12):
        print(epoch)
        for batch in range(1,6):
            trainX,trainY=ld.getbatch(batch)
            if(epoch%2==0):
                learningRate= min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch/decay_speed)

            trainData={X:trainX,Y:trainY,lr:learningRate,pkeep: 0.75}
            sess.run(train_step,feed_dict=trainData)



    time2 = (datetime.datetime.time(datetime.datetime.now()))
    print('the start time is  :  ',time2)
    print("The Accuracy on train data is : ",accuracy.eval(feed_dict=testData,session=sess))



if __name__=="__main__":
    ConvolutionlNetwork()