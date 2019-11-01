import os
import tensorflow as tf
import pickle
import numpy as np

input_img_feature_dir = "./data/bb/"
out_dir =os.path.join('runs', "transfer")


class ImageCaptionData():
    def __init__(self, img_feature_dir, valid_percent=0.1, test_percent=0.1, is_shuffle=True):
        self._all_img_feature_filepaths = []
        for filename in os.listdir(img_feature_dir):
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
            with open(filepath, 'rb') as f:
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


def main():
    n_class = 5
    learning_rate = 0.001
    batch_size = 100
    training_step = 10000
    checkpoint_every = 100  # 每隔checkpoint_every保存一次模型和测试摘要

    # 加载文件
    caption_data = ImageCaptionData(input_img_feature_dir, valid_percent=0.1, test_percent=0.1)
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
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 保存检查点
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=6)

        for i in range(training_step):  # steps = 10000
            # 每次获取一个batch的训练数据
            batch_data, batch_labels = caption_data.train_next(batch_size)
            sess.run(optimizer, feed_dict={x_transfer: batch_data, y_transfer: batch_labels})
            # 在验证集上测试正确率
            if i % 100 == 0 or (i + 1) == training_step:
                batch_data, batch_labels = caption_data._valid_data, caption_data._valid_labels
                validation_accuracy = sess.run(accuracy, feed_dict={x_transfer: batch_data, y_transfer: batch_labels})
                print('Step %d : Validation accuracy  = %.1f%%' % (i, validation_accuracy * 100))

            if i % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print('Saved model checkpoint to {}\n'.format(path))

        # 最后在测试集上测试正确率
        test_data, test_labels = caption_data._test_data, caption_data._test_labels
        test_accuracy = sess.run(accuracy, {x_transfer: test_data, y_transfer: test_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()
