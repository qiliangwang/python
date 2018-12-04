import tensorflow as tf

'''
Tensor("inputs:0", shape=(?, 128, 128, 3), dtype=float32)
Tensor("decoder/conv2d/Relu:0", shape=(?, 64, 64, 64), dtype=float32)
Tensor("decoder/conv2d_1/Relu:0", shape=(?, 32, 32, 128), dtype=float32)
Tensor("decoder/conv2d_2/Relu:0", shape=(?, 16, 16, 256), dtype=float32)
Tensor("decoder/conv2d_3/Relu:0", shape=(?, 8, 8, 512), dtype=float32)
Tensor("decoder/conv2d_4/Relu:0", shape=(?, 4, 4, 1024), dtype=float32)
Tensor("decoder/dense/Relu:0", shape=(?, 2048), dtype=float32)
Tensor("decoder/dense_1/Relu:0", shape=(?, 1024), dtype=float32)
Tensor("encoder/dense/Relu:0", shape=(?, 2048), dtype=float32)
Tensor("encoder/Reshape:0", shape=(?, 4, 4, 1024), dtype=float32)
Tensor("encoder/conv2d_transpose/Relu:0", shape=(?, 8, 8, 512), dtype=float32)
Tensor("encoder/conv2d_transpose_1/Relu:0", shape=(?, 16, 16, 256), dtype=float32)
Tensor("encoder/conv2d_transpose_2/Relu:0", shape=(?, 32, 32, 128), dtype=float32)
Tensor("encoder/conv2d_transpose_3/Relu:0", shape=(?, 64, 64, 64), dtype=float32)
Tensor("encoder/conv2d_transpose_4/Relu:0", shape=(?, 128, 128, 3), dtype=float32)
'''


class ConvolutionAutoEncoder(object):
    def __init__(self, input_dim, out_dim, lr, drop_out):
        self.inputs = tf.placeholder(tf.float32, [None, *input_dim], name='inputs')
        self.outputs = tf.placeholder(tf.float32, [None, *out_dim], name='outputs')
        self.lr = lr
        self.drop = drop_out
        self.compressed = self.decoder()
        self.logits, self.decode = self.encoder()
        self.loss = self.create_loss()
        self.opt = self.optimizer()
        pass

    def decoder(self):
        with tf.variable_scope('decoder'):
            print(self.inputs)
            x1 = tf.layers.conv2d(self.inputs, 64, kernel_size=[5, 5], strides=[2, 2], padding='same',
                                  )
            print(x1)

            x2 = tf.layers.conv2d(x1, 128, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x2)

            x3 = tf.layers.conv2d(x2, 256, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x3)

            x4 = tf.layers.conv2d(x3, 512, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x4)

            x5 = tf.layers.conv2d(x4, 1024, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.tanh)
            print(x5)

            # x6 = tf.layers.dense(tf.layers.flatten(x5), 2048, activation=tf.nn.relu)
            # print(x6)

            # x7 = tf.layers.dense(x6, 1024, activation=tf.nn.relu)
            # print(x7)

            return x5
        pass

    def encoder(self):
        with tf.variable_scope('encoder'):

            # x6 = tf.layers.dense(self.compressed, 2048, activation=tf.nn.relu)
            # print(x6)

            # x5 = tf.reshape(tf.layers.dense(self.compressed, 1024*4*4, activation=tf.nn.relu), (-1, 4, 4, 1024))
            # print(x5)

            x4 = tf.layers.conv2d_transpose(self.compressed, 512, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x4)

            x3 = tf.layers.conv2d_transpose(x4, 256, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x3)

            x2 = tf.layers.conv2d_transpose(x3, 128, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x2)

            x1 = tf.layers.conv2d_transpose(x2, 64, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(x1)

            logits = tf.layers.conv2d_transpose(x1, 3, kernel_size=[5, 5], strides=[2, 2], padding='same')

            decode = tf.nn.sigmoid(logits, name='decode')

            return logits, decode

    def create_loss(self):
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.outputs, logits=self.logits)
        loss = tf.reduce_mean(cost)
        return loss

    def optimizer(self):
        return tf.train.AdamOptimizer(0.0002).minimize(self.loss)


def main():
    model = ConvolutionAutoEncoder((128, 128, 3), (128, 128, 3), 0.1, 0.7)
    pass


if __name__ == '__main__':
    main()
