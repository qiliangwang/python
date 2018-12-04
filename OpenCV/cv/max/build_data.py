import cv2 as cv
import os
import numpy as np


def plot_img(images):
    image = np.zeros([512, 512, 3])
    for height in range(4):
        for width in range(4):
            image[height*128:height*128+128, width*128:width*128+128, :] = images[4*height+width]
    return image


def batch_iter(data, labels, batch_size, shuffle=True):
    data_size = len(data)
    data = np.array(data)
    labels = np.array(labels)
    num_batches = ((data_size-1)//batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = data[shuffle_indices]
        shuffled_y = labels[shuffle_indices]
    else:
        shuffled_x = data
        shuffled_y = labels
    for batch_num in range(num_batches):
        start_index = batch_num*batch_size
        end_index = min((batch_num+1)*batch_size, data_size)
        yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]


def parse_img(path, num, dim):
    img = cv.imread(path)
    img = img / 255.0
    images = []
    len_x = img.shape[0]//num
    for height in range(num):
        for width in range(num):
            temp_img = img[height * len_x:height * len_x + len_x, width * len_x:width * len_x + len_x, :]
            # print(temp_img.shape)
            temp_img = cv.resize(temp_img, (dim, dim))
            temp_img = temp_img.reshape(-1, dim, dim, 3)
            images.append(temp_img)
    return images


def convert_2_img(paths, num, base_path):
    images = []
    for path in paths:
        path = os.path.join(base_path, path)
        img = parse_img(path, num, 128)
        images.extend(img)
    images = np.concatenate(images)
    # print(images.shape)
    return images


def main():
    data_dir = '../data'
    img_128_dirs = os.listdir(os.path.join(data_dir, 'photo128'))
    img_256_dirs = os.listdir(os.path.join(data_dir, 'photo256'))
    img_512_dirs = os.listdir(os.path.join(data_dir, 'photo512'))

    for batch_x, batch_y in batch_iter(img_256_dirs, img_512_dirs, batch_size=4):
        y = convert_2_img(batch_y, 4, os.path.join(data_dir, 'photo512'))
        x = convert_2_img(batch_x, 4, os.path.join(data_dir, 'photo256'))
        print(y.shape)
        print(x.shape)
        # demo_y = y[0].reshape(128, 128, 3)
        # demo_x = x[0].reshape(128, 128, 3)
        # cv.imshow('x', demo_x)
        # cv.imshow('y', demo_y)
        # cv.waitKey(0)
        # break

        print('==========>')


if __name__ == '__main__':
    main()
