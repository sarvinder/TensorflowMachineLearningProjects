###This file contains the Simple Five layer neural network that will help to recoganize the handwriten digits
###The data is comming from the mnist dataset which contain the examples of hand writen digits

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def NeuralNetwork():
    # neural network with 5 layers
    #
    # · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28 hidden layer 1
    # \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
    #  · · · · · · · · ·                                                h1 [batch, 200]    hidden layer 2
    #   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
    #    · · · · · · ·                                                  Y2 [batch, 100]    hidden layer 3
    #     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
    #      · · · · ·                                                    Y3 [batch, 60]     hidden layer 4
    #       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
    #        · · ·                                                      Y4 [batch, 30]     hidden layer 5
    #         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
    #          ·                                                        Y5 [batch, 10]     output layer (hypothesis layer)

    """Hyper Parameter and Some others"""
    learningRate=0.03
    batchSize=100
    Epochs=10000
    logsPathTrain = "./logs/train"
    logsPathTest = "./logs/test"


    """Get the data (DATA)"""
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True,reshape=False,validation_size=0)


    tf.reset_default_graph()

    """Five layer neurons"""
    L = 200
    M = 100
    N = 60
    O = 30

    """This is the first input layer passed as an image"""
    with tf.name_scope("Input"):
        X=tf.placeholder(tf.float32,shape=[None,28,28,1],name="X_input")

    with tf.name_scope("labels"):
        Y=tf.placeholder(tf.float32,shape=[None,10],name="Y_labels")

    """In Neural network the weights are initialized randomly"""
    with tf.name_scope("hidden_layers_weights_and_biases"):
        W1 = tf.Variable(tf.truncated_normal([784,L],stddev=0.1))
        b1 = tf.Variable(tf.zeros([L]))
        W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        b2 = tf.Variable(tf.zeros([M]))
        W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        b3 = tf.Variable(tf.zeros([N]))
        W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        b4 = tf.Variable(tf.zeros([O]))
        W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
        b5 = tf.Variable(tf.zeros([10]))

    """Our NEURAL NETWORK Model"""
    with tf.name_scope("model"):
        h1=tf.reshape(X,[-1,784])
        h2=tf.nn.relu(tf.matmul(h1,W1)+b1)
        h3=tf.nn.relu(tf.matmul(h2,W2)+b2)
        h4=tf.nn.relu(tf.matmul(h3,W3)+b3)
        h5=tf.nn.relu(tf.matmul(h4,W4)+b4)
        ylogits=tf.matmul(h5,W5)+b5
        hx=tf.nn.softmax(ylogits)

    with tf.name_scope("cost_function"):
        """This is the Cross entropy function"""
        Jtheta=tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx),reduction_indices=[1]))


    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(hx, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("optimizer"):
        train_step=tf.train.GradientDescentOptimizer(learningRate).minimize(Jtheta)

    tf.summary.scalar("cost",Jtheta)
    tf.summary.scalar("accuracy",accuracy)
    summary_op = tf.summary.merge_all()

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    """write the graph summary to the file"""
    writer=tf.summary.FileWriter(logsPathTrain,graph=tf.get_default_graph())

    writer1=tf.summary.FileWriter(logsPathTest,graph=tf.get_default_graph())

    testData = {X: mnist.test.images, Y: mnist.test.labels}
    for epoch in range(Epochs):
        batch_X,batch_Y=mnist.train.next_batch(batchSize)
        trainData={X:batch_X,Y:batch_Y}
        sess.run(train_step,feed_dict=trainData)
        _,c,summary=sess.run([train_step,Jtheta,summary_op],feed_dict=trainData)
        writer.add_summary(summary,epoch)
        if(epoch%5==0):
            c, summary = sess.run([ Jtheta, summary_op], feed_dict=testData)
            writer1.add_summary(summary, epoch)

    print(    "Accuracy on test: ", accuracy.eval(feed_dict=testData,session=sess))
    sess.close()



if __name__=="__main__":
    NeuralNetwork()

