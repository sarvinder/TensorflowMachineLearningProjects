import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
#******************************************************************************************
#***********NeuralNetwork with Regularization and decaying learning rate*******************
#***********With the AdamOptimizer and this will be a Five layer network*******************
#******************************************************************************************

def trainAndTest():
    #************************
    batchSize=100
    epochs=10000
    logsPathTrain = "./logs/train"
    logsPathTest = "./logs/test"
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    tf.reset_default_graph()

    """Five layer neurons"""
    L = 200
    M = 100
    N = 60
    O = 30

    """Get the data"""
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True,reshape=False,validation_size=0)

    with tf.name_scope('input'):
        X=tf.placeholder(tf.float32,shape=[None,28,28,1],name="X_Label")
        Y=tf.placeholder(tf.float32,shape=[None,10],name="Y_Label")

    with tf.name_scope("learningRateDecay"):
        lr=tf.placeholder(tf.float32)

    with tf.name_scope("probToKeepNode"):
        pKeep=tf.placeholder(tf.float32)

    """The weights and biases wiil be equal to the number of layers we want"""
    with tf.name_scope("WeightsAndBiases"):
        W1=tf.Variable(tf.truncated_normal([784,L],stddev=0.1))
        b1=tf.Variable(tf.zeros([L]))
        W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        b2 = tf.Variable(tf.zeros([M]))
        W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        b3 = tf.Variable(tf.zeros([N]))
        W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        b4 = tf.Variable(tf.zeros([O]))
        W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
        b5 = tf.Variable(tf.zeros([10]))

    with tf.name_scope("model"):
        h1=tf.reshape(X,[-1,784])

        h2=tf.nn.relu(tf.matmul(h1,W1)+b1)
        h2d=tf.nn.dropout(h2,pKeep)

        h3=tf.nn.relu(tf.matmul(h2d,W2)+b2)
        h3d = tf.nn.dropout(h3, pKeep)

        h4=tf.nn.relu(tf.matmul(h3d,W3)+b3)
        h4d = tf.nn.dropout(h4, pKeep)

        h5=tf.nn.relu(tf.matmul(h4d,W4)+b4)
        h5d = tf.nn.dropout(h5, pKeep)

        ylogits=tf.matmul(h5d,W5)+b5
        hx=tf.nn.softmax(ylogits)

    with tf.name_scope("accuracy"):
        correct_prediction=tf.equal(tf.argmax(hx,1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.name_scope("CostFunction"):
        JTheta=tf.nn.softmax_cross_entropy_with_logits(logits=ylogits,labels=Y)
        JTheta=tf.reduce_mean(JTheta)*100

    with tf.name_scope("optimizer"):
        train_step=tf.train.AdamOptimizer(lr).minimize(JTheta)

    tf.summary.scalar("cost", JTheta)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    """Start the training with the session"""
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    learningRate=max_learning_rate
    """write the graph summary to the file"""
    writer = tf.summary.FileWriter(logsPathTrain, graph=tf.get_default_graph())
    writer1 = tf.summary.FileWriter(logsPathTest, graph=tf.get_default_graph())

    testData = {X: mnist.test.images, Y: mnist.test.labels,pKeep:1.0}

    print("the Training Starts")
    for epoch in range(epochs):
        trainX,trainY=mnist.train.next_batch(batchSize)
        trainData1={X:trainX,Y:trainY,pKeep:1.0}
        if(epoch%2==0):
            learningRate= min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch/decay_speed)
        trainData={X:trainX,Y:trainY,lr:learningRate,pKeep: 0.75}
        sess.run(train_step,feed_dict=trainData)
        c, summary = sess.run([ JTheta, summary_op], feed_dict=trainData1)
        writer.add_summary(summary, epoch)
        if (epoch % 5 == 0):
            c, summary = sess.run([JTheta, summary_op], feed_dict=testData)
            writer1.add_summary(summary, epoch)

    print("The Accuracy on train data is : ",accuracy.eval(feed_dict=testData,session=sess))

if __name__=="__main__":
    trainAndTest()