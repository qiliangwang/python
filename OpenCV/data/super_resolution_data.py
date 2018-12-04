import os
import cv2 as cv
import numpy as np
import random
from sklearn.metrics import confusion_matrix


class SuperResolutionData:
    def __init__(self, data_dir='D:\\chest_xray'):
        self.test_ratio = 0.2
        # self.label_ratio = 0.3
        self.train_ratio = 0.1
        # self.pool = Pool()
        self.data_dir = data_dir
        train_set = os.path.join(self.data_dir, 'train')
        self.train_normal_dir = os.path.join(train_set, 'NORMAL')
        self.train_pneumonia_dir = os.path.join(train_set, 'PNEUMONIA')

        test_set = os.path.join(self.data_dir, 'test')
        self.test_normal_dir = os.path.join(test_set, 'NORMAL')
        self.test_pneumonia_dir = os.path.join(test_set, 'PNEUMONIA')

        # 自己设定的没有label的数目
        self.label_normal = 23*20
        self.label_pneumonia = 39*20

        self.train_normal, self.train_pneumonia, self.test_normal, self.test_pneumonia, self.none_label_data = self.build_data_v2()

        self.train_normal_labels, self.train_pneumonia_labels, self.test_normal_labels, self.test_pneumonia_labels, self.none_label_labels = self.build_labels()

        self.dim = 128

        self.true_labels = None

        self.predict_labels = None
        pass

    def build_data(self):
        # 使用train set的部分数据作为train set
        # 部分数据作为 none label  test 的数据作为none label
        test_normal = os.listdir(self.test_normal_dir)
        test_normal = [os.path.join(self.test_normal_dir, _) for _ in test_normal]

        test_pneumonia = os.listdir(self.test_pneumonia_dir)
        test_pneumonia = [os.path.join(self.test_pneumonia_dir, _) for _ in test_pneumonia]

        # 使用训练集的部分数据作为没有标签的数据，部分数据作为test集
        train_normal_full = os.listdir(self.train_normal_dir)
        train_normal_full = [os.path.join(self.train_normal_dir, _) for _ in train_normal_full]

        train_pneumonia_full = os.listdir(self.train_pneumonia_dir)
        train_pneumonia_full = [os.path.join(self.train_pneumonia_dir, _) for _ in train_pneumonia_full]

        # none_label_normal = train_normal_full[self.label_normal:]
        # none_label_pneumonia = train_normal_full[self.label_pneumonia:]
        # 得到全部的data, 分别shuffle 然后 split shuffle_indices = np.random.permutation(np.arange(data_size))
        normal = test_normal + train_normal_full
        random.shuffle(normal)
        # normal = normal
        # print('Normal:', len(normal))
        """
        Normal: 1575
        Pneumonia: 4265
        """
        pneumonia = test_pneumonia + train_pneumonia_full
        random.shuffle(pneumonia)
        # start
        test_normal_size = len(normal) * self.test_ratio
        test_pneumonia_size = len(pneumonia) * self.test_ratio

        # print('Pneumonia:', len(pneumonia))
        none_label_data = train_normal_full + train_normal_full + test_normal + test_pneumonia
        # none_label_data = train_normal + train_pneumonia + test_normal_full + test_pneumonia_full

        # train_normal = train_normal_full[:self.label_normal]
        # train_pneumonia = train_pneumonia_full[:self.label_pneumonia]

        train_normal = train_normal_full
        train_pneumonia = train_pneumonia_full

        return train_normal, train_pneumonia, test_normal, test_pneumonia, none_label_data
        pass

    def build_data_v2(self):
        # 使用train set的部分数据作为train set
        # 部分数据作为 none label  test 的数据作为none label
        test_normal = os.listdir(self.test_normal_dir)
        test_normal = [os.path.join(self.test_normal_dir, _) for _ in test_normal]

        test_pneumonia = os.listdir(self.test_pneumonia_dir)
        test_pneumonia = [os.path.join(self.test_pneumonia_dir, _) for _ in test_pneumonia]

        # 使用训练集的部分数据作为没有标签的数据，部分数据作为test集
        train_normal_full = os.listdir(self.train_normal_dir)
        train_normal_full = [os.path.join(self.train_normal_dir, _) for _ in train_normal_full]

        train_pneumonia_full = os.listdir(self.train_pneumonia_dir)
        train_pneumonia_full = [os.path.join(self.train_pneumonia_dir, _) for _ in train_pneumonia_full]

        # none_label_normal = train_normal_full[self.label_normal:]
        # none_label_pneumonia = train_normal_full[self.label_pneumonia:]
        # 得到全部的data, 分别shuffle 然后 split shuffle_indices = np.random.permutation(np.arange(data_size))
        normal = test_normal + train_normal_full
        random.shuffle(normal)

        pneumonia = test_pneumonia + train_pneumonia_full
        random.shuffle(pneumonia)
        """
               Normal: 1575
               Pneumonia: 4265
        """
        # start
        test_normal_size = int(len(normal) * self.test_ratio)
        test_pneumonia_size = int(len(pneumonia) * self.test_ratio)

        train_normal_size = int(len(normal) * self.train_ratio)
        train_pneumonia_size = int(len(pneumonia) * self.train_ratio)

        test_normal = normal[: test_normal_size]
        test_pneumonia = pneumonia[: test_pneumonia_size]
        print('Test Info : normal', len(test_normal), 'pneumonia', len(test_pneumonia))

        train_normal = normal[test_normal_size: test_normal_size + train_normal_size]
        train_pneumonia = pneumonia[test_pneumonia_size: test_pneumonia_size + train_pneumonia_size]
        print('Train Info : normal', len(train_normal), 'pneumonia', len(train_pneumonia))

        none_label_data = normal[test_normal_size + train_normal_size:] + pneumonia[test_pneumonia_size + train_normal_size:] + train_normal + train_pneumonia
        print('None label Info: ', len(none_label_data))

        return train_normal, train_pneumonia, test_normal, test_pneumonia, none_label_data

    def build_labels(self):
        # 0 normal 1 pneumonia
        # [1, 0] for normal, [0, 1] for pneumonia [0, 0] for none label data
        train_normal_labels = [[1, 0] for _ in self.train_normal]

        train_pneumonia_labels = [[0, 1] for _ in self.train_pneumonia]

        test_normal_labels = [[1, 0] for _ in self.test_normal]

        test_pneumonia_labels = [[0, 1] for _ in self.test_pneumonia]

        none_label_labels = [[0, 0] for _ in self.none_label_data]

        return train_normal_labels, train_pneumonia_labels, test_normal_labels, test_pneumonia_labels, none_label_labels
        pass

    @staticmethod
    def convert_img(img_dirs):

        def parse_img(img_dir):
            img = cv.imread(img_dir) / 255.0
            img = cv.resize(img, (128, 128))
            img = img.reshape(-1, 128, 128, 3)
            return img

        images = [parse_img(img_dir) for img_dir in img_dirs]
        return np.concatenate(images)

    def batch_iter(self, data_type, batch_size, shuffle=True):

        data = np.array(self.none_label_data)
        labels = np.array(self.none_label_labels)

        if data_type == 'train':

            data = np.array(self.train_normal + self.train_pneumonia)
            labels = np.array(self.train_normal_labels + self.train_pneumonia_labels)

        if data_type == 'test':

            data = np.array(self.test_normal + self.test_pneumonia)
            labels = np.array(self.test_normal_labels + self.test_pneumonia_labels)

        # shuffle data
        # print('Data: ', data.shape)
        # print('Labels', labels.shape)
        data_size = len(data)
        num_batches = ((data_size-1)//batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_x = data[shuffle_indices]
            shuffle_y = labels[shuffle_indices]
        else:
            shuffle_x = data
            shuffle_y = labels
        # 0 normal 1 pneumonia
        # [1, 0] for normal, [0, 1] for pneumonia [0, 0] for none label data
        # [[0] if label == [1] for label in shuffle_y]
        # print('=================>')
        # demo = [list(label) for label in shuffle_y]
        self.true_labels = [[0] if list(label) == [1, 0] else [1] for label in shuffle_y]
        # print(self.true_labels)
        # print(demo)
        # print('<=================')
        # []
        # print(shuffle_y)
        for batch_num in range(num_batches):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1) * batch_size, data_size)

            batch_x = self.convert_img(shuffle_x[start_index:end_index])
            batch_y = shuffle_y[start_index:end_index]

            yield batch_x, batch_y
        pass

    def cal_labels(self, predict_results):
        """
        R 语言里面，好像直接调一个函数confusionMatrix(predictions$Class_predicted, predictions$Class_actual, positive = 'Pneumonia') 就可以跑出一堆指标
        指标 1、accuracy 2、precision； 3 recall （recall = sensitivity = true positive rate） ；   4、 specificity； 5  F-score
        :return:
        """
        true_labels = np.array(self.true_labels)
        predict_results = predict_results.reshape(-1, 1)
        cnf_matrix = confusion_matrix(true_labels, predict_results)
        print(cnf_matrix)
        return cnf_matrix
        pass

    # @staticmethod

        # plt.show()


def main():
    data = MedicineData()
    for e in range(20):
        for batch_x, batch_y in data.batch_iter(data_type='test', batch_size=1):
            # print(batch_x.shape, batch_y.shape)
            # print(e)
            pass
    pass


def vader(batch_size):
    """
    just for fun
    :return:
    """
    data = [i for i in range(500)]
    data_size = len(data)
    num_batches = ((data_size - 1) // batch_size) + 1
    # for e in range(20):
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
    pass


class MongodbData:
    def __init__(self):
        pass

    pass


if __name__ == '__main__':
    # for v in vader(batch_size=50):
    #     print(v)
    main()
