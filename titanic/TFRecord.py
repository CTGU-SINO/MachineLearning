import os
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# 制作TFRecord格式
def createTFRecord(filename,label,data):
    writer = tf.python_io.TFRecordWriter(filename)
    for x,y in zip(label,data):
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(x),
                'data': _float_feature(y)
            }))
            writer.write(example.SerializeToString())
    writer.close()

# 读取train.tfrecord中的数据
def read_and_decode(filename,num_epochs,shuffle):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], shuffle=shuffle,num_epochs=num_epochs)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'data': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['data'], tf.uint8)
    img = tf.reshape(img, [20, 40, 7])
    labels = tf.cast(features['label'], tf.int32)
    return img, labels


def createBatch(filename, batch_size,shuffle,num_epochs=None):
    images, labels = read_and_decode(filename,num_epochs,shuffle)

    min_train_examples = 15000
    min_test_examples = 10000

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=batch_size,
                                                      capacity=min_train_examples + 3 * batch_size,
                                                      min_after_dequeue=min_train_examples
                                                      )
    else:
        image_batch, label_batch = tf.train.batch(
            [images, labels],
            batch_size=batch_size,
            capacity=min_test_examples + 3 * batch_size)

    #label_batch = tf.one_hot(label_batch, depth=2)
    tf.summary.image('images', image_batch)
    return image_batch, label_batch