from models.AutoEncoder import ConvolutionAutoEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', validation_size=0)


def sample(reconstructed, noisy_images, epoch):

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))

    for images, row in zip([noisy_images, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.savefig('../results/denoise/'+str(epoch)+'.jpg')
    pass


def main():
    model = ConvolutionAutoEncoder((28, 28, 1), (28, 28, 1), 0.001)
    sess = tf.InteractiveSession()
    epochs = 200
    batch_size = 200
    noise_factor = 0.5
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)
            imgs = batch[0].reshape((-1, 28, 28, 1))

            # Add random noise
            noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)

            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            batch_loss, _ = sess.run([model.loss, model.opt], feed_dict={model.inputs: noisy_imgs,
                                                             model.targets: imgs})

            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Training loss: {:.4f}".format(batch_loss))
        #     test model
        test_images = mnist.test.images[:10]
        noisy_images = test_images + noise_factor * np.random.randn(*test_images.shape)
        noisy_images = np.clip(noisy_images, 0., 1.)

        reconstructed = sess.run(model.decoded, feed_dict={model.inputs: noisy_images.reshape((10, 28, 28, 1))})
        sample(reconstructed, noisy_images, e)

    pass


if __name__ == '__main__':
    main()
