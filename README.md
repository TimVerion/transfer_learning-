# transfer_learning(精简)

首先将tensorflow_inception_graph.pb放置在model/tensorflow_inception_graph.pb

1.首先使用parse_img.py 对自己的数据集进行检查
注意这里的input_img_dir="自己图片集合的位置"
change_name() 方法可以将文件中图片的名字进行集体更改
2.对检查好的图片进行特征的提取feature_extraction.py
设置好自己图片文件夹位置：input_img_dir=""
输出路径：output_folder=""
3.训练自己的模型，training_model.py使用第2步中的特征文件 XXXXX.pickle
模型存放的位置：
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', "transfer"))
4.使用训练好的模型进行测试：show_model.py
再  ./data/test 中放置16张测试图片
图片名字的格式  ：  类别名字_XXX.jpg
改成自己类别的名字,注意和os.listdir()
agricultures = ["corn", "millet", "rice", "sorghum", "wheat"]