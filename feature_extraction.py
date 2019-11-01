import os
import tensorflow as tf
from tensorflow import gfile  # 负责读取pb文件,构造图
import pickle
import numpy as np

# 使用inception v3 来对图像进行提取特征
# 对于checkpoint存的各个变量的数值,使用restore加载
# 除了上述的,还可以使用freeze_graph将生成的模型合并成一个pb文件
# 这样使用就不用重构原来的图
model_file = "./model/tensorflow_inception_graph.pb"
input_img_dir = "C:/Users/Tim/Desktop/imgs"
output_folder = "./data/bb/"

batch_size = 500

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def parse__file(file):
    img_names = []
    labels = []
    for i, j in enumerate(os.listdir(file)):
        img_dir = os.listdir(os.path.join(file, j))
        img_names += [os.path.join(j, i) for i in img_dir]
        labels += [i for _ in range(len(img_dir))]
    return np.array(img_names), np.array(labels)


def load_pretrained_inception_v3(model_file):
    with gfile.FastGFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()  # 构造一个空的图
        graph_def.ParseFromString(f.read())  # 将计算图读取进来
        _ = tf.import_graph_def(graph_def, name="")  # 将图导入到默认图


all_img_names, all_img_labels = parse__file(input_img_dir)
load_pretrained_inception_v3(model_file)

num_batches = int(len(all_img_names) / batch_size)  # 计算每批次batch_size个共有多少个子文件
if len(all_img_names) % batch_size != 0:
    num_batches += 1  # 将多余的文件使用一个batch
with tf.Session() as sess:
    second_to_last_tensor = sess.graph.get_tensor_by_name(
        "pool_3/_reshape:0")  # Tensor("pool_3/_reshape:0", shape=(1, 2048), dtype=float32)
    # dev_summary_writer = tf.summary.FileWriter(r'C:\Users\Tim\Desktop\transfer_learning(精简)\summary',graph=sess.graph)
    # exit()
    for i in range(num_batches):
        batch_img_names = all_img_names[i * batch_size: (i + 1) * batch_size]  # 按批量去除name
        batch_labels = all_img_labels[i * batch_size: (i + 1) * batch_size]  # 按批量去除name
        batch_features = []
        for img_name in batch_img_names:
            img_path = os.path.join(input_img_dir, img_name)
            if not os.path.exists(img_path):
                raise Exception("%s doesn't exists" % img_path)
            img_data = gfile.FastGFile(img_path, "rb").read()  # 读取图片
            feature_vector = sess.run(second_to_last_tensor, feed_dict={"DecodeJpeg/contents:0": img_data})
            batch_features.append(feature_vector)
        batch_features = np.vstack(batch_features)
        output_filename = os.path.join(output_folder, "image_features-%d.pickle" % i)
        print(output_filename, "....ok......")
        with gfile.GFile(output_filename, 'w') as f:  # 打开一个文件
            pickle.dump((batch_img_names, batch_features, batch_labels), f)  # 将数据保存在文件中
    #     # 图像预处理完成
