import tensorflow as tf
import re

INITIALIZER_FULLY = tf.contrib.layers.xavier_initializer()
INITIALIZER_CON2D = tf.contrib.layers.xavier_initializer_conv2d()
BATCH_SIZE = 128

IMAGE_SIZE = 224
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, initializer, wd):
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_op(x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=INITIALIZER_CON2D)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)

        _activation_summary(z)
    return z


def res_block_layers(x, name, n_out_list, change_dimension=False, block_stride=1):
    if change_dimension:
        short_cut_conv = conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                                 dh=block_stride, dw=block_stride,
                                 padding="SAME", activation=None)
    else:
        short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3, kw=3,
                           dh=1, dw=1,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_3 = conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                           dh=1, dw=1,
                           padding="SAME", activation=None)

    block_res = tf.add(short_cut_conv, block_conv_3)
    res = tf.nn.relu(block_res)
    return res



def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2,padding="SAME"):
    return tf.nn.max_pool(x,ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def avg_pool_op(x, name, kh=2, kw=2, dh=2, dw=2,padding="SAME"):
    return tf.nn.avg_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=INITIALIZER_FULLY)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        _activation_summary(fc)

    return fc

def inference(X):

    # ResNet


    usBN = True
    training = True
    conv1 = conv_op(X, "conv1", 64, training, usBN, 3, 3, 1, 1)
    pool1 = max_pool_op(conv1, "pool1", kh=3, kw=3)

    block1_1 = res_block_layers(pool1, "block1_1", [64, 256], True, 1)
    block1_2 = res_block_layers(block1_1, "block1_2", [64, 256], False, 1)
    block1_3 = res_block_layers(block1_2, "block1_3", [64, 256], False, 1)

    block2_1 = res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
    block2_2 = res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
    block2_3 = res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
    block2_4 = res_block_layers(block2_3, "block2_4", [128, 512], False, 1)

    block3_1 = res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
    block3_2 = res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
    block3_3 = res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
    block3_4 = res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
    block3_5 = res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
    block3_6 = res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)

    block4_1 = res_block_layers(block3_6, "block4_1", [512, 2048], True, 2)
    block4_2 = res_block_layers(block4_1, "block4_2", [512, 2048], False, 1)
    block4_3 = res_block_layers(block4_2, "block4_3", [512, 2048], False, 1)

    pool2 = avg_pool_op(block4_3, "pool2", kh=7, kw=7, dh=1, dw=1, padding="SAME")
    shape = pool2.get_shape()
    fc_in = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])
    logits = fc_op(fc_in, "fc1", NUM_CLASSES)

    return logits

    # CNN


    # with tf.variable_scope('conv1') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 3, 64],
    #                                          initializer=INITIALIZER_CON2D,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #     _activation_summary(conv1)
    #
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                        padding='SAME', name='pool1')
    #
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')
    #
    # with tf.variable_scope('conv2') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 64, 64],
    #                                          initializer=INITIALIZER_CON2D,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #     _activation_summary(conv2)
    #
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    #
    # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    # with tf.variable_scope('local3') as scope:
    #     reshape = tf.reshape(pool2, [X.get_shape().as_list()[0], -1])
    #     dim = reshape.get_shape()[1].value
    #     weights = _variable_with_weight_decay('weights', shape=[dim, 384],
    #                                           initializer=INITIALIZER_FULLY, wd=0.004)
    #     biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #     local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #     _activation_summary(local3)
    #
    # with tf.variable_scope('local4') as scope:
    #     weights = _variable_with_weight_decay('weights', shape=[384, 192],
    #                                           initializer=INITIALIZER_FULLY, wd=0.004)
    #     biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    #     local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    #     _activation_summary(local4)
    #
    # with tf.variable_scope('softmax_linear') as scope:
    #     weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
    #                                           initializer=INITIALIZER_FULLY, wd=None)
    #     biases = _variable_on_cpu('biases', [NUM_CLASSES],
    #                               tf.constant_initializer(0.0))
    #     softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #     _activation_summary(softmax_linear)
    #
    # return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, k=1)
    return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')