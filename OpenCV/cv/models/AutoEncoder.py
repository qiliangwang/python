import tensorflow as tf

'''
Tensor("inputs:0", shape=(?, 28, 28, 1), dtype=float32)
Tensor("decoder/conv2d_1/BiasAdd:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("decoder/conv2d_2/Relu:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("decoder/conv2d_3/BiasAdd:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("decoder/conv2d_4/Relu:0", shape=(?, 14, 14, 16), dtype=float32)
Tensor("decoder/conv2d_5/BiasAdd:0", shape=(?, 14, 14, 16), dtype=float32)
Tensor("encoder/ResizeNearestNeighbor:0", shape=(?, 7, 7, 16), dtype=float32)
Tensor("encoder/conv2d/Relu:0", shape=(?, 7, 7, 16), dtype=float32)
Tensor("encoder/ResizeNearestNeighbor_1:0", shape=(?, 14, 14, 16), dtype=float32)
Tensor("encoder/conv2d_1/Relu:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("encoder/ResizeNearestNeighbor_2:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("encoder/conv2d_2/Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("encoder/decoded:0", shape=(?, 28, 28, 1), dtype=float32)
'''


def log(*x):
    print(*x)
    pass


class ConvolutionAutoEncoder(object):
    def __init__(self, input_dim, out_dim, lr):
        self.inputs = tf.placeholder(tf.float32, [None, *input_dim], name='inputs')
        '(28, 28, 1) here'
        self.targets = tf.placeholder(tf.float32, [None, *out_dim], name='outputs')
        self.lr = lr
        # self.drop = drop_out
        self.encoded = self.encoder()
        self.decoded, self.loss, self.opt = self.decoder()

    def encoder(self):
        with tf.name_scope('decoder'):
            log(self.inputs)
            # @log
            convolution_1 = tf.layers.conv2d(self.inputs, 32, (3, 3), padding='same', activation=tf.nn.relu)

            max_pool_1 = tf.layers.conv2d(convolution_1, 32, kernel_size=(2, 2), strides=(2, 2), padding='valid')
            log(max_pool_1)

            convolution_2 = tf.layers.conv2d(max_pool_1, 32, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_2)

            max_pool_2 = tf.layers.conv2d(convolution_2, 32, kernel_size=(2, 2), strides=(2, 2), padding='valid')
            # max_pool_2 = tf.layers.max_pooling2d(convolution_2, (2, 2), (2, 2), padding='same')
            log(max_pool_2)

            convolution_3 = tf.layers.conv2d(max_pool_2, 16, (3, 3), padding='same', activation=tf.nn.relu)
            log('convolution_3', convolution_3)
            # Now 7x7x16
            encoded = tf.layers.conv2d(convolution_3, 16, kernel_size=(2, 2), strides=(1, 1), padding='same')
            # encoded = tf.layers.max_pooling2d(convolution_3, (2, 2), (2, 2), padding='same')
            log('encoded', encoded)

            return encoded

    def decoder(self):
        with tf.name_scope('encoder'):

            up_sample_1 = tf.layers.conv2d(self.encoded, 16, kernel_size=(2, 2), strides=(1, 1), padding='same')

            log('up_sample_1', up_sample_1)

            convolution_4 = tf.layers.conv2d(up_sample_1, 16, (2, 2), padding='same', activation=tf.nn.relu)
            log(convolution_4)

            up_sample_2 = tf.layers.conv2d_transpose(convolution_4, 16, kernel_size=(3, 3), strides=[2, 2], padding='same')

            log(up_sample_2)

            convolution_5 = tf.layers.conv2d(up_sample_2, 32, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_5)

            up_sample_3 = tf.layers.conv2d_transpose(convolution_5, 32, kernel_size=(3, 3), strides=[2, 2], padding='same')
            log(up_sample_3)

            convolution_6 = tf.layers.conv2d(up_sample_3, 32, (3, 3), padding='same', activation=tf.nn.relu)
            log(convolution_6)

            logits = tf.layers.conv2d(convolution_6, 1, (3, 3), padding='same', activation=None)

            decoded = tf.nn.sigmoid(logits, name='decoded')
            log(decoded)

            cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits)

            loss = tf.reduce_mean(cost)

            opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

            return decoded, loss, opt


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
    model = ConvolutionAutoEncoder((28, 28, 1), (28, 28, 1), 0.001)
    sess = tf.InteractiveSession()

    epochs = 100
    batch_size = 50
    # Set's how much noise we're adding to the MNIST images
    noise_factor = 0.5
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)
            # Get images from the batch
            imgs = batch[0].reshape((-1, 28, 28, 1))

            # Add random noise to the input images
            noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            # Noisy images as inputs, original images as targets
            batch_cost, _ = sess.run([model.loss, model.opt], feed_dict={model.inputs: noisy_imgs,
                                                                         model.targets: imgs})

            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Training loss: {:.4f}".format(batch_cost))

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
        in_imgs = mnist.test.images[:10]
        noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        reconstructed = sess.run(model.decoded, feed_dict={model.inputs: noisy_imgs.reshape((10, 28, 28, 1))})

        for images, row in zip([noisy_imgs, reconstructed], axes):
            for img, ax in zip(images, row):
                ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.1)
        plt.savefig('../results/denoise/' + str(e) + '.jpg')
    # model = ConvolutionAutoEncoder((28, 28, 1), (28, 28, 1), 0.001)
    pass


if __name__ == '__main__':
    main()
