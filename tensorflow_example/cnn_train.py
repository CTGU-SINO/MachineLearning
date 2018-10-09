import tensorflow as tf
import cnn_input
import datetime
import cnn
import os

TRAIN_DIR = 'train.TFRecords'
OUT_DIR = 'out'
BATCH_SIZE = 128
MAX_STEPS = 1000000
CHECKPOINT_FREQUENCY = 1000
SESSION_CONF = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=True,
                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))


def train():
    with tf.Graph().as_default():
        sess = tf.Session(config=SESSION_CONF)
        with sess.as_default():

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.device('/cpu:0'):
                images, labels = cnn_input.createBatch(TRAIN_DIR, BATCH_SIZE, True)
            logits = cnn.inference(images)
            loss = cnn.loss(logits, labels)
            train_op = cnn.train(loss, global_step)
            accuracy = cnn.evaluation(logits, labels)
            summary = tf.summary.merge_all()

            checkpoint_dir = os.path.abspath(os.path.join(OUT_DIR, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            summary_writer = tf.summary.FileWriter(OUT_DIR + "/summary", sess.graph)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            tf.train.start_queue_runners(sess=sess)

            for step in range(MAX_STEPS):
                _, step, cur_loss, cur_acc = sess.run([train_op, global_step, loss, accuracy])
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))

                if step % 100 == 0:
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                cur_step = tf.train.global_step(sess, global_step)

                if cur_step % CHECKPOINT_FREQUENCY == 0 and cur_step != 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                    print("Saved model checkpoint to {}\n".format(path))


train()
