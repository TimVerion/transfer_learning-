import os
import tensorflow as tf
from tensorflow import gfile
import pickle
import numpy as np
import cv2

model_file = "./model/tensorflow_inception_graph.pb"
input_img_feature_dir = "./data/bb/"
out_dir =os.path.join('runs', "transfer")
CHECKPOINT_DIR = os.path.join(out_dir, 'checkpoints')
agricultures = ["corn", "millet", "rice", "sorghum", "wheat"]

class ImageCaptionData():
    def __init__(self, img_feature_dir, valid_percent=0.1, test_percent=0.1, is_shuffle=True):
        self._all_img_feature_filepaths = []
        for filename in gfile.ListDirectory(img_feature_dir):
            self._all_img_feature_filepaths.append(os.path.join(img_feature_dir, filename))
        self.is_shuffle = is_shuffle
        self._img_feature_data = []
        self._img_feature_labels = []
        self._load_img_feature_pickle()
        if self.is_shuffle:
            self._random_shuffle()
        # 根据数据集加载出训练，验证，测试集
        valid_len = int(self.size() * valid_percent)
        test_len = int(self.size() * test_percent)
        self._valid_data = self._img_feature_data[:valid_len, :]
        self._valid_labels = self._img_feature_labels[:valid_len]
        self._test_data = self._img_feature_data[valid_len:valid_len + test_len, :]
        self._test_labels = self._img_feature_labels[valid_len:valid_len + test_len]
        self._train_data = self._img_feature_data[valid_len + test_len:, :]
        self._train_labels = self._img_feature_labels[valid_len + test_len:]
        self._train_indicator = 0
        print(self._valid_data.shape, self._valid_labels.shape)
        print(self._test_data.shape, self._test_labels.shape)
        print(self._train_data.shape, self._train_labels.shape)

    def _load_img_feature_pickle(self):
        for filepath in self._all_img_feature_filepaths:
            print("loading %s" % filepath)
            with gfile.GFile(filepath, 'rb') as f:
                _, features, labels = pickle.load(f)
                self._img_feature_data.append(features)
                self._img_feature_labels += [labels]
        self._img_feature_data = np.vstack(self._img_feature_data)
        self._img_feature_labels = np.hstack(self._img_feature_labels)
        print(self._img_feature_data.shape)
        print(self._img_feature_labels.shape)

    def size(self):
        return len(self._img_feature_labels)

    def _random_shuffle(self):
        p = np.random.permutation(self.size())
        self._img_feature_data = self._img_feature_data[p]
        self._img_feature_labels = self._img_feature_labels[p]

    def train_next(self, batch_size):
        end_indicator = self._train_indicator + batch_size
        if end_indicator > len(self._train_labels):
            p = np.random.permutation(len(self._train_labels))
            self._train_data = self._train_data[p]
            self._train_labels = self._train_labels[p]
            self._train_indicator = 0
            end_indicator = self._train_indicator + batch_size
        assert end_indicator <= len(self._train_labels)
        batch_img_features = self._train_data[self._train_indicator: end_indicator]
        batch_img_labels = self._train_labels[self._train_indicator: end_indicator]
        self._train_indicator = end_indicator
        return batch_img_features, batch_img_labels


def load_pretrained_inception_v3(model_file):
    with gfile.FastGFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()  # 构造一个空的图
        graph_def.ParseFromString(f.read())  # 将计算图读取进来
        _ = tf.import_graph_def(graph_def, name="")  # 将图导入到默认图


def img_to_vector(base_dir):
    with tf.Session() as sess:
        second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3/_reshape:0")
        batch_features = []
        test_data = []
        test_labels = []
        for i in os.listdir(base_dir):
            img = cv2.imread(os.path.join(base_dir, i))
            img = cv2.resize(img, (448, 448))  # 重置
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转通道
            test_data.append(img)
            test_labels.append(i.split("_")[0])
            img_data = gfile.FastGFile(os.path.join(base_dir, i), "rb").read()  # 读取图片
            feature_vector = sess.run(second_to_last_tensor, feed_dict={"DecodeJpeg/contents:0": img_data})
            batch_features.append(feature_vector)
        batch_features = np.vstack(batch_features)
    return batch_features, test_data, test_labels


def show_img(test_data, test_labels, pre_labels):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 4)
    for i, axi in enumerate(ax.flat):
        axi.imshow(test_data[i])
        print(test_labels[i], pre_labels[i])
        axi.set_xlabel(xlabel=pre_labels[i], color='black' if test_labels[i] == pre_labels[i] else 'red')
        axi.set(xticks=[], yticks=[])
    plt.show()


def main():
    load_pretrained_inception_v3(model_file)
    test_bottlenecks, test_data, test_labels = img_to_vector("./data/test/")
    n_class = 5
    learning_rate = 0.001
    # 占位符X:以图片特征向量作为输入
    x_transfer = tf.placeholder(tf.float32, [None, 2048])
    # 占位符Y：[None,5] 5表示5种类的农作品
    y_transfer = tf.placeholder(tf.int64, [None])  # [None,5]
    # 定义一层全连接层解决新的图片分类问题
    fc_1 = tf.layers.dense(x_transfer, 1024)
    fc_2 = tf.layers.dense(fc_1, n_class)
    final_tensor = tf.nn.softmax(fc_2)

    # 定义交叉熵损失函数
    loss_transfer = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_2, labels=y_transfer))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_transfer)

    # 计算正确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_tensor, 1), y_transfer), tf.float32))

    # 训练过程
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        saver.restore(sess, checkpoint_file)
        test_demo = sess.run(tf.argmax(fc_2, 1), feed_dict={x_transfer: test_bottlenecks})
        pre_labels = [agricultures[k] for k in test_demo]
        show_img(test_data, test_labels, pre_labels)


if __name__ == '__main__':
    main()
