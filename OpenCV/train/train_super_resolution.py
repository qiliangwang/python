import tensorflow as tf
from models.super_resolution_model import SuperResolutionNet
from data.super_resolution_data import SuperResolutionData


def main():
    model = SuperResolutionNet((128, 128, 3), (128, 128, 3), 0.1, 0.7)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    epochs = 20
    data = SuperResolutionData()
    count = 0
    for batch_x, batch_y in data.build_data(epochs, 2):
        model_dict = {model.inputs: batch_x, model.outputs: batch_y}
        sess.run([model.opt], feed_dict=model_dict)
        count += 1
        if count % 20 == 0:
            loss, decode = sess.run([model.loss, model.decode], feed_dict=model_dict)
            data.plot_img(decode[:16], count // 20)
            print('Loss', loss)


if __name__ == '__main__':
    main()
