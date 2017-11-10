import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import datetime


def ConvolutionNeuralNetwork():
    tf.reset_default_graph()
    time1 = (datetime.datetime.time(datetime.datetime.now()))
    print('the start time is  :  ',time1)

    mnist=input_data.read_data_sets('MNIST_data',one_hot=True,reshape=False,validation_size=0)

    with tf.name_scope("Input"):
        # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
        X=tf.placeholder(tf.float32,[None,28,28,1])
        # correct answers will go here
        Y=tf.placeholder(tf.float32,[None,10])
        # variable learning rate
        lr=tf.placeholder(tf.float32)
        # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
        pkeep=tf.placeholder(tf.float32)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (the last layer has 10 softmax neurons)
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layer

    ##hyperParameters
    epochs=12000
    batchSize=100
    logsPathTrain = "./logs/train"
    logsPathTest = "./logs/test"
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0

    with tf.name_scope("weightsAndBiases"):
        W1=tf.Variable(tf.truncated_normal([6,6,1,K],stddev=0.1))##will
        b1=tf.Variable(tf.constant(0.1,tf.float32,[K]))

        W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))  ##will
        b2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

        W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))  ##will
        b3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

        W4 = tf.Variable(tf.truncated_normal([7* 7* M, N], stddev=0.1))  ##will
        b4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

        W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))  ##will
        b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

    with tf.name_scope("Model"):
        stride=1#output 28*28
        hx1=tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,stride,stride,1],padding='SAME')+b1)

        stride = 2#output 14*14
        hx2 = tf.nn.relu(tf.nn.conv2d(hx1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)

        stride = 2#output 7*7
        hx3 = tf.nn.relu(tf.nn.conv2d(hx2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)

        reshape=tf.reshape(hx3,[-1,7*7*M])

        hx4=tf.nn.relu(tf.matmul(reshape,W4)+b4)
        hx4d=tf.nn.dropout(hx4,pkeep)
        ylogits=tf.matmul(hx4d,W5)+b5
        hx=tf.nn.softmax(ylogits)

    with tf.name_scope("costFunction"):
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=ylogits,labels=Y)
        cross_entropy=tf.reduce_mean(cross_entropy)*100

    with tf.name_scope("accuracy"):
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(hx, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("optimizer"):
        train_step=tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    writer = tf.summary.FileWriter(logsPathTrain, graph=tf.get_default_graph())
    writer1 = tf.summary.FileWriter(logsPathTest, graph=tf.get_default_graph())


    learning_rate =max_learning_rate

    testData = {X: mnist.test.images, Y: mnist.test.labels,pkeep:1.0}

    print("the Training Starts")
    for epoch in range(epochs):
        print(epoch)
        trainX,trainY=mnist.train.next_batch(batchSize)
        trainData1={X:trainX,Y:trainY,pkeep:1.0}
        if(epoch%2==0):
            learningRate= min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch/decay_speed)
        trainData={X:trainX,Y:trainY,lr:learningRate,pkeep: 0.75}
        sess.run(train_step,feed_dict=trainData)
        c, summary = sess.run([ cross_entropy, summary_op], feed_dict=trainData1)
        writer.add_summary(summary, epoch)
        if (epoch % 5 == 0):
            c, summary = sess.run([cross_entropy, summary_op], feed_dict=testData)
            writer1.add_summary(summary, epoch)


    time2 = (datetime.datetime.time(datetime.datetime.now()))
    print('the start time is  :  ',time2)
    print("The Accuracy on train data is : ",accuracy.eval(feed_dict=testData,session=sess))




if __name__=="__main__":
    ConvolutionNeuralNetwork()