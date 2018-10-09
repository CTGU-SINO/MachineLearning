import tensorflow as tf


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)# also tf.float32 implicitly
print(node1, node2)


with tf.Session() as sess1:
    print(sess1.run([node1, node2]))


with tf.Session() as sess2:
    node3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess2.run(node3))