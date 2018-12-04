import tensorflow as tf


def log(x):
    print(x)
    pass


class ConvolutionAutoEncoder(object):
    def __init__(self, input_dim, out_dim, lr):
        self.inputs = tf.placeholder(tf.float32, [None, *input_dim], name='inputs')
        '(28, 28, 1) here'
        self.targets = tf.placeholder(tf.float32, [None, *out_dim], name='outputs')
        self.lr = lr
        self.encoded = self.encoder()
        self.decoded, self.loss, self.opt = self.decoder()

    def encoder(self):
        with tf.name_scope('encoder'):
            log(self.inputs)
            # @log
            convolution_1 = tf.layers.conv2d(self.inputs, 32, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_1)

            max_pool_1 = tf.layers.max_pooling2d(convolution_1, (2, 2), (2, 2), padding='same')
            log(max_pool_1)

            convolution_2 = tf.layers.conv2d(max_pool_1, 32, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_2)

            max_pool_2 = tf.layers.max_pooling2d(convolution_2, (2, 2), (2, 2), padding='same')
            log(max_pool_2)

            convolution_3 = tf.layers.conv2d(max_pool_2, 16, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_3)
            # Now 7x7x16
            encoded = tf.layers.max_pooling2d(convolution_3, (2, 2), (2, 2), padding='same')
            log(encoded)

            return encoded
        pass

    def decoder(self):
        with tf.name_scope('decoder'):
            # tf.VariableScope

            up_sample_1 = tf.image.resize_nearest_neighbor(self.encoded, (7, 7))
            # log(up_sample_1)

            convolution_4 = tf.layers.conv2d(up_sample_1, 16, (3, 3), padding='same', activation=tf.nn.relu)
            # log(convolution_4)

            up_sample_2 = tf.image.resize_nearest_neighbor(convolution_4, (14, 14))
            # log(up_sample_2)

            convolution_5 = tf.layers.conv2d(up_sample_2, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # log(convolution_5)

            up_sample_3 = tf.image.resize_nearest_neighbor(convolution_5, (28, 28))
            # log(up_sample_3)

            convolution_6 = tf.layers.conv2d(up_sample_3, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # log(convolution_6)

            logits = tf.layers.conv2d(convolution_6, 1, (3, 3), padding='same', activation=None)

            decoded = tf.nn.sigmoid(logits, name='decoded')
            # log(decoded)

            cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits)

            loss = tf.reduce_mean(cost)

            opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

            return decoded, loss, opt


def main():
    model = ConvolutionAutoEncoder((28, 28, 1), (28, 28, 1), 0.001)
    pass


if __name__ == '__main__':
    main()
