import tensorflow as tf
import numpy as np
import pandas as pd
import re

np.set_printoptions(threshold=np.NaN)
pd.set_option('display.max_rows', None)

train = pd.read_csv('titanic/input/train.csv', header=0, dtype={'Age': np.float64})
test = pd.read_csv('titanic/input/test.csv', header=0, dtype={'Age': np.float64})

full_data = [train, test]

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)

train = train.values
test = test.values


def one_hot_encoding(data):
    data_map = {
        1: np.array([0, 1]),
        0: np.array([1, 0])
    }
    return [data_map[x] for x in data]


X = (train[0::, 1::])[2:]
y = (train[0::, 0])[2:]
one_hot_trainLabel = np.mat(one_hot_encoding(y))


class Dataset:
    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    def next_batch(self, batchSize, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batchSize > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batchSize - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batchSize
            end = self._index_in_epoch
            return self._data[start:end]


tf.set_random_seed(1)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 20

n_inputs = 7  # data input (data shape: 1*7)
n_steps = 1  # time steps
n_hidden_units = 32  # neurons in hidden layer
n_classes = 2  # classes (2)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (7, 32)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (32, 2)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (32, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (2, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (20 batch * 40 steps, 7 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (20 batch * 40 steps, 32 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (20 batch, 40 steps, 32 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (32, 2)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        train_x, train_y = Dataset(X).next_batch(batch_size), Dataset(one_hot_trainLabel).next_batch(batch_size)
        train_x = train_x.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: train_x,
            y: train_y
        })
        print(sess.run(accuracy, feed_dict={
            x: train_x,
            y: train_y
        }))
        step += 1
