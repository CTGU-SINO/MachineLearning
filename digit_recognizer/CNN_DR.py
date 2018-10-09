import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
from os.path import isfile


def W_variable(name, shape):
    return tf.get_variable(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer_conv2d())


def B_variable(name, shape):
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(0.0))


def conV2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling(input):
    return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def preprocessing(data):
    minV = 0
    maxV = 255
    data = (data - minV) / (maxV - minV)
    return data


def one_hot_encoding(data, numberOfClass):
    lb = LabelBinarizer()
    lb.fit(range(numberOfClass))
    return lb.transform(data)


def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch + 1, numberOfEpoch),
          '\tTraining Loss: {:.3f}'.format(trainLoss),
          '\tValidation Loss: {:.3f}'.format(validationLoss),
          '\tAccuracy: {:.2f}%'.format(validationAccuracy * 100))


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

trainData = train.values[:, 1:]
trainLabel = train.values[:, 0]
testData = test.values

processedTrainData = preprocessing(trainData)
processedTestData = preprocessing(testData)
one_hot_trainLabel = one_hot_encoding(trainLabel, 10)

fileName = 'mnist.p'
if not isfile(fileName):
    pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData), open(fileName, 'wb'))

# load pickle file
fileName = 'mnist.p'
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode='rb'))

graph = tf.Graph()
tf.reset_default_graph()
with graph.as_default():
    xs = tf.placeholder(tf.float32, (None, 784))
    ys = tf.placeholder(tf.float32, (None, 10))
    keep_problem = tf.placeholder(tf.float32)
    # CNN1
    c1 = conV2d(tf.reshape(xs, [-1, 28, 28, 1]), W_variable('w1', [5, 5, 1, 32]))  # 28*28*32
    r1 = tf.nn.relu(c1 + B_variable('b1', [32]))
    max_pooling1 = max_pooling(r1)  # 14*14*32

    # CNN2
    c2 = conV2d(max_pooling1, W_variable('w2', [5, 5, 32, 64]))  # 14*14*64
    r2 = tf.nn.relu(c2 + B_variable('b2', [64]))
    max_pooling2 = max_pooling(r2)  # 7*7*64

    # fulling connection
    f1_input = tf.reshape(max_pooling2, [-1, 7 * 7 * 64])
    f1 = tf.nn.relu(tf.matmul(f1_input, W_variable('f_w', [7 * 7 * 64, 1024]) + B_variable('f_b', [1024])))
    d1 = tf.nn.dropout(f1, keep_problem)

    sw = W_variable('s_w', [1024, 10])
    sb = B_variable('b_w', [10])
    prediction = tf.nn.softmax(tf.matmul(d1, sw) + sb)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # accuracy
    correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

save_dir = './save'
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(20):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(trainData, one_hot_trainLabel, test_size=0.2)
        # training loss
        for i in range(0, len(train_x), 64):
            trainLoss, _, _ = sess.run([cross_entropy, prediction, train_step], feed_dict={
                xs: train_x[i: i + 64],
                ys: train_y[i: i + 64],
                keep_problem: 1.0
            })

            # validation loss
            valAcc, valLoss = sess.run([accuracy, cross_entropy], feed_dict={
                xs: val_x,
                ys: val_y,
                keep_problem: 1.0
            })

            # print out
            printResult(epoch, 20, trainLoss, valLoss, valAcc)
    # save
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
