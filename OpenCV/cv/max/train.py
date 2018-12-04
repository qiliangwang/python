import tensorflow as tf
from max.build_data import *
from max.AutoEncoderOne import ConvolutionAutoEncoder
import time


def main():
    data_dir = '../data'
    # img_128_dirs = os.listdir(os.path.join(data_dir, 'photo128'))
    img_256_dirs = os.listdir(os.path.join(data_dir, 'photo256'))
    img_512_dirs = os.listdir(os.path.join(data_dir, 'photo512'))
    epochs = 100
    model = ConvolutionAutoEncoder((128, 128, 3), (128, 128, 3), 0.1, 0.7)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    count = 0
    for epoch in range(epochs):
        for batch_x, batch_y in batch_iter(img_256_dirs, img_512_dirs, batch_size=4):
            y = convert_2_img(batch_y, 4, os.path.join(data_dir, 'photo512'))
            x = convert_2_img(batch_x, 4, os.path.join(data_dir, 'photo256'))
            model_dict = {model.inputs: x, model.outputs: y}
            count = count + 1
            # run opt
            sess.run([model.opt], feed_dict=model_dict)
            # _, loss = sess.run([model.opt, model.loss], feed_dict=model_dict)
            if count % 20 == 0:
                loss, decode = sess.run([model.loss, model.decode], feed_dict=model_dict)
                image_decode = plot_img(decode[:16])
                image_decode = image_decode * 255
                cv.imwrite('../data/results3/' + str(count//20) + '.jpg', image_decode)
                print('Epoch:', epoch, 'Loss', loss)


if __name__ == '__main__':
    main()
