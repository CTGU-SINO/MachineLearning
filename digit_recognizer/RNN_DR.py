import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

n_steps = 28
n_inputs = 28

n_outputs = 10
n_neurons = 150

graph = tf.Graph()
tf.reset_default_graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs], name="X")
    Y = tf.placeholder(tf.int64, shape=(None), name="Y")

    basic_Cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    outputs1, states = tf.nn.dynamic_rnn(basic_Cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs)
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    print("loss")

    learning_rate = 0.0001
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    print("train")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean((tf.cast(correct, tf.float32)))
    print("eval")

    saver = tf.train.Saver()

print("init")

dataset = pd.read_csv("input/train.csv")
X_data = dataset.iloc[:, 1:785].values
Y_data = dataset.iloc[:, 0:1].values
X_train, X_valid, y_train, y_valid = train_test_split(X_data, Y_data, test_size=0.2)
print("input")

X_valid = X_valid.reshape((-1, n_steps, n_inputs))
y_train = y_train.reshape(33600, )
y_valid = y_valid.reshape(8400, )
n_epochs = 300
batch_size = 20
lengthofdata = len(X_train)
print("Initialized")

save_dir = './save'
list1 = []
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", graph)
    for epochs in range(n_epochs):
        i = 0
        while i < lengthofdata:
            start = i
            end = i + batch_size
            X_batch = np.array(X_train[start:end])
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))

            Y_batch = np.array(y_train[start:end])
            i = end
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
        acc_test = accuracy.eval(feed_dict={X: X_valid, Y: y_valid})
        plt.plot(acc_test)
        print(epochs, "Train Accuracy: ", acc_train, "Test Accuracy", acc_test)
        list1.append(acc_test)
    save_path = saver.save(sess, "./save/my_model_final.ckpt")
