import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
import mindspore
from mindspore import dataset
from src.configs import *

DATA_PATH = 'data/MNIST_DATA'

# train set
train_images_file = os.path.join(DATA_PATH, 'train-images.idx3-ubyte') 
# train labels set
train_labels_file = os.path.join(DATA_PATH, 'train-labels.idx1-ubyte') 
# test set
test_images_file = os.path.join(DATA_PATH, 't10k-images.idx3-ubyte')
# test labels set
test_labels_file = os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte')


"""解析图片数据"""
def decode_idx3_ubyte(idx3_ubyte_file):
    # get the binary date
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic numbers :%d, image numbers: %d, image size: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    # parse date set
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i1 in range(num_images):
        if (i1 + 1) % 10000 == 0:
            print('parsed %d' % (i1 + 1) + 'images')
            print(offset)
        images[i1] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

"""解析标签数据"""
def decode_idx1_ubyte(idx1_ubyte_file):
    # get the binary date
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic numbers :%d, image numbers : %d' % (magic_number, num_images))
    # parse date set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i2 in range(num_images):
        labels[i2] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

class DatasetGenerator:
    def __init__(self):
        self.data = decode_idx3_ubyte(train_images_file).astype("float32")
        self.label = decode_idx1_ubyte(train_labels_file).astype("float32")
    def __getitem__(self, item):
        return self.data[item], self.label[item]
    def __len__(self):
        return len(self.data)

class DatasetGenerator_valid:
    def __init__(self):
        self.data =decode_idx3_ubyte(train_images_file).astype("float32")
        self.label = decode_idx1_ubyte(train_labels_file).astype("float32")
    def __getitem__(self, item):
        return self.data[item], self.label[item]
    def __len__(self):
        return len(self.data)


# 创建训练数据集,GAN网络只需要训练数据即可
def create_dataset_train(batch_size=64, repeat_size=1, latent_size=100):
    dataset_generator = DatasetGenerator()
    dataset1 = dataset.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
    # 数据映射操作
    mnist_ds = dataset1.map(
        operations=lambda x: (
            x.astype("float32"), # 未归一化
            np.random.normal(size=(latent_size)).astype("float32")
        ),
        output_columns=["image", "latent_code"],
        column_order=["image", "latent_code"]
    )
    # 批量操作
    mnist_ds = mnist_ds.batch(batch_size, True)
    # mnist_ds = mnist_ds.repeat(1)  # 数据加倍
    return mnist_ds

# 创建验证数据集
def create_dataset_valid(batch_size=64, repeat_size=1, latent_size=100):
    dataset_generator = DatasetGenerator_valid()
    dataset = dataset.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False)
    mnist_ds = dataset.map(
        operations=lambda x: (
            x[-10000:].astype("float32"),
            np.random.normal(size=(latent_size)).astype("float32")
        ),
        output_columns=["image", "latent_code"],
        column_order=["image", "latent_code"]
    )
    # 批量操作
    mnist_ds = mnist_ds.batch(batch_size, True)
    mnist_ds = mnist_ds.repeat(1)
    return mnist_ds

# 获取处理后的数据集
dataset = create_dataset_train(batch_size=BATCH_SIZE, repeat_size=1, latent_size=latent_size)
print(type(dataset))
# 获取数据集大小
iter_size = dataset.get_dataset_size()
print('iter size is %d' % (iter_size))




# 可视化部分训练数据
src_data = './result/src_data.png'

# 可视化部分训练数据
def visualize(dataset):
    data_iter = next(dataset.create_dict_iterator(output_numpy=True))
    figure = plt.figure(figsize=(5, 5))
    cols, rows = 5, 5
    for idx in range(1, cols * rows + 1):
        image = data_iter['image'][idx]
        figure.add_subplot(rows, cols, idx)
        plt.axis("off")
        plt.imshow(image.squeeze(), cmap="gray")
    plt.savefig(src_data)
    plt.show()
    
visualize(dataset)