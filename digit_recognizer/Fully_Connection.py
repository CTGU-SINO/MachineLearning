import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import cv2

initializer_fully = tf.contrib.layers.xavier_initializer()
TOTAL_EPOCH = 30
INPUT_SIZE = 784
FULLY_NODE1 = 1600
FULLY_NODE2 = 800
FULLY_NODE3 = 100
OUTPUT_SIZE = 10
BATCH_SIZE = 128
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 1000.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 1e-4

np.set_printoptions(threshold=np.nan)


def W_variable(name, shape, wd=None):
    var = tf.get_variable(name, shape, initializer=initializer_fully)
    if not wd is None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(wd)(var))
    return var


def B_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def bn(x, beta_name, gamma_name, name="bn"):
    axes = [d for d in range(len(x.get_shape()))]
    beta = tf.get_variable(beta_name, shape=[], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable(gamma_name, shape=[], initializer=tf.constant_initializer(1.0))
    x_mean, x_variance = tf.nn.moments(x, axes)
    y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, 1e-10, name)
    return y


# load pickle file
fileName = 'mnist.p'
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(
    open(fileName, mode='rb'))


def model(action):
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='data')
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name='label')
        with tf.name_scope('Fully_Connection1'):
            w1 = W_variable('f_w1', [INPUT_SIZE, FULLY_NODE1], 0.001)
            b1 = B_variable('f_b1', [FULLY_NODE1])
            f1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            d1 = tf.nn.dropout(f1, 1.0)
        with tf.name_scope('Fully_Connection2'):
            w2 = W_variable('f_w2', [FULLY_NODE1, FULLY_NODE2], 0.001)
            b2 = B_variable('f_b2', [FULLY_NODE2])
            f2 = tf.nn.relu(tf.matmul(d1, w2) + b2)
            d2 = tf.nn.dropout(f2, 0.5)
        with tf.name_scope('Fully_Connection3'):
            w3 = W_variable('f_w3', [FULLY_NODE2, FULLY_NODE3], 0.001)
            b3 = B_variable('f_b3', [FULLY_NODE3])
            f3 = tf.nn.relu(tf.matmul(d2, w3) + b3)
            d3 = bn(f3, 'beta3', 'gamma3', 'batch_normalization3')
            # d3 = tf.nn.dropout(f3, 0.5)
        with tf.name_scope('SoftMax'):
            sw = W_variable('s_w', [FULLY_NODE3, OUTPUT_SIZE])
            sb = B_variable('s_b', [OUTPUT_SIZE])
            prediction = tf.nn.softmax(tf.matmul(d3, sw) + sb)
        with tf.name_scope("loss"):
            loss = -tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1])
            cross_entropy = tf.reduce_mean(loss, name='loss')
        with tf.name_scope("train"):
            decay_steps = int(50 * NUM_EPOCHS_PER_DECAY)  # 多少步衰减
            global_step = tf.Variable(0)

            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            training_op = optimizer.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope("eval"):
            correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    best_predict = 0
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        checkpoint_dir = os.path.abspath(os.path.join('fully', "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        if action == 'train':
            tf.summary.merge_all()
            tf.summary.FileWriter("fully/summary", graph)
            for epoch in range(TOTAL_EPOCH):
                train_x, val_x, train_y, val_y = train_test_split(
                    trainData, one_hot_trainLabel, test_size=0.2)
                for i in range(0, len(train_x), BATCH_SIZE):
                    trainLoss, _, _ = sess.run([cross_entropy, prediction, training_op], feed_dict={
                        x: train_x[i: i + BATCH_SIZE],
                        y: train_y[i: i + BATCH_SIZE]
                    })
                    predict, _ = sess.run([accuracy, cross_entropy], feed_dict={
                        x: train_x[i: i + BATCH_SIZE],
                        y: train_y[i: i + BATCH_SIZE]
                    })
                    # print("Epoch: {}/{},Training Loss: {:.3f},Validation Loss: {:.3f},Accuracy: {:.2f}%"
                    #       .format(epoch + 1, 20,trainLoss,validationLoss,validationAccuracy * 100))
                    print("Epoch: {}/{},Training Loss: {:.3f},Predict: {}"
                          .format(epoch + 1, TOTAL_EPOCH, trainLoss, predict))
                valid_predict, _ = sess.run([accuracy, cross_entropy], feed_dict={
                    x: val_x,
                    y: val_y
                })
                print("Epoch: {}/{},Val_Predict: {}".format(epoch + 1, TOTAL_EPOCH, valid_predict))
                if valid_predict > best_predict:
                    best_predict = valid_predict
                    saver.save(sess, checkpoint_prefix)
        elif action == 'test':
            result_list = []
            for test in testData:
                y_ = sess.run([prediction], feed_dict={x: [test]})
                result = np.argmax(y_)
                result_list.append(result)
            sample = pd.read_csv('input/sample_submission.csv')
            sample.Label = result_list
            sample.to_csv("fully_connection_submission.csv", index=False)


if __name__ == '__main__':
    model('test')
