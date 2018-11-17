import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# LOAD DATA
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# BUILD NETWORK
# inputs
inputs = tf.placeholder(tf.float32, [None, 784])
input_images = tf.reshape(inputs, [-1, 28, 28, 1])
targets = tf.placeholder(tf.int32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)

# first conv_pool
conv_one = tf.layers.conv2d(input_images,
                            filters=32,
                            kernel_size=[5, 5],
                            strides=[1, 1],
                            padding='same')
max_pool_one = tf.layers.max_pooling2d(conv_one,
                                       pool_size=[2, 2],
                                       strides=[2, 2])
print(conv_one.shape)
print(max_pool_one.shape)
"""
(?, 28, 28, 32)
(?, 14, 14, 32)
"""
# second conv_pool
conv_two = tf.layers.conv2d(max_pool_one,
                            filters=64,
                            kernel_size=[2, 2],
                            strides=[1, 1],
                            padding='same')
max_pool_two = tf.layers.max_pooling2d(conv_two,
                                       pool_size=[2, 2],
                                       strides=[2, 2])
print(conv_two.shape)
print(max_pool_two.shape)
'''
(?, 14, 14, 64)
(?, 7, 7, 64)
'''
flatten_layer = tf.layers.flatten(max_pool_two)
print(flatten_layer.shape)
'''
(?, 3136)
'''
# fc1
fc1 = tf.layers.dense(flatten_layer,
                      1024,
                      activation=tf.nn.relu)
fc1 = tf.layers.dropout(fc1,
                        rate=dropout_rate)
# fc2
logits = tf.layers.dense(fc1, 10)
out = tf.nn.softmax(logits)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                            logits=logits))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
# acc
correct_predicts = tf.equal(tf.argmax(out, 1),
                            tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32))

# exit()
# training

batch_size = 50
epochs = 20
n_batches = mnist.train.num_examples//batch_size
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print('Initialized')
for epoch in range(1, epochs+1):
    for batch in range(n_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_feed = {inputs: batch_x,
                      targets: batch_y,
                      dropout_rate: 0.5}
        sess.run(optimizer, feed_dict=train_feed)
    val_feed = {inputs: mnist.test.images,
                targets: mnist.test.labels,
                dropout_rate: 1.0}
    acc = sess.run(accuracy, feed_dict=val_feed)
    print("Epoch: {} Accuracy: {}".format(epoch, acc))
sess.close()
