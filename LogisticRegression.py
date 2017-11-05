import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def logisticModel():

    """for the graph"""
    tf.reset_default_graph()

    """Set the learning Rate,training time,batch size ,the path where the graph data will be stored"""
    learningRate=0.3
    Epochs=1500
    batchSize=100
    logsPathTrain="./logs/train"##This is where the data for the graph will be stored
    logsPathTest="./logs/test"

    """Get the data (DATA)"""
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    """build the tensorflow graph"""
    # ----X is the input (IMAGES)
    # ----Y is the Lables
    # ----W is for the weights(theta)
    # ----b is the bias vector
    with tf.name_scope('input'):
        X=tf.placeholder(tf.float32,shape=[None,784],name="X_input")
        Y=tf.placeholder(tf.float32,shape=[None,10],name="Y_input")

    with tf.name_scope('weights'):
        W=tf.Variable(tf.zeros([784,10]))

    with tf.name_scope("biases"):
        b=tf.Variable(tf.zeros([10]))

    """define the hx function which will be used in cost"""
    with tf.name_scope("softmax"):
        hx=tf.nn.softmax(tf.matmul(X,W)+b)

    """The Cost function"""
    with tf.name_scope("cross_entropy"):
        Jtheta=tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx),reduction_indices=[1]))

    """The optimizer to update the Weights"""
    with tf.name_scope("optimizer"):
        train_step=tf.train.GradientDescentOptimizer(learningRate).minimize(Jtheta)

    """The Accuracy function"""
    with tf.name_scope("accuracy"):
        correctPrediction=tf.equal(tf.argmax(hx,1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correctPrediction,tf.float32))

    """Adding to the scallar graph"""
    tf.summary.scalar("cost",Jtheta)
    tf.summary.scalar("accuracy",accuracy)
    #tf.summary.scalar("weights", W)
    #tf.summary.scalar("biases", b)
    summary_op = tf.summary.merge_all()

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    """write the graph summary to the file"""
    writer=tf.summary.FileWriter(logsPathTrain,graph=tf.get_default_graph())

    writer1=tf.summary.FileWriter(logsPathTest,graph=tf.get_default_graph())

    testData = {X: mnist.test.images, Y: mnist.test.labels}
    """START THE TRAINING STEP"""
    for epoch in range(Epochs):
        batch_X,batch_Y=mnist.train.next_batch(batchSize)
        trainData={X:batch_X,Y:batch_Y}
        _,c,summary=sess.run([train_step,Jtheta,summary_op],feed_dict=trainData)
        writer.add_summary(summary,epoch)
        if(epoch%5==0):
            c, summary = sess.run([ Jtheta, summary_op], feed_dict=testData)
            writer1.add_summary(summary, epoch)

    print(    "Accuracy on test: ", accuracy.eval(feed_dict=testData,session=sess))
    sess.close()

if __name__=="__main__":
    logisticModel()