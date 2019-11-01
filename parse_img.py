import os
import numpy as np

# 自己找到的文件夹位置
input_img_dir = "C:/Users/Tim/Desktop/imgs"
import cv2


def change_name():
    # 根据文件夹命名 : apple/234423.png   --> apple/apple_001.jpg
    names = os.listdir(input_img_dir)
    for name_dir in names:
        base_dir = os.listdir(os.path.join(input_img_dir, name_dir))
        for i, name in enumerate(base_dir):
            file_name = os.path.join(input_img_dir, name_dir, name)
            num = "%03d" % i
            new_name = os.path.join(input_img_dir, name_dir, name_dir + "_" + str(num) + ".jpg")
            os.rename(file_name, new_name)


def parse__file(file):
    img_names = []
    labels = []
    for i, j in enumerate(os.listdir(file)):
        img_dir = os.listdir(os.path.join(file, j))
        img_names += [os.path.join(j, i) for i in img_dir]
        labels += [i for _ in range(len(img_dir))]
    return np.array(img_names), np.array(labels)


# 测试图片是否可用
def img_test_ok(all_img_names, is_write):
    for f in all_img_names:
        print(f)
        img_data = cv2.imread(os.path.join(input_img_dir, f), 1)
        if is_write:
            cv2.imwrite(os.path.join(input_img_dir, f), img_data)
        if img_data.shape[-1] == 4:
            print("error")
    print("数据没问题...")


# 将图片的名字规范化
# change_name()
# 测试图片是不是有问题
all_img_names, all_img_labels = parse__file(input_img_dir)
img_test_ok(all_img_names, False)
print(all_img_names.shape, all_img_labels.shape)
