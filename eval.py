import gc
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mindspore import context
import mindspore.numpy
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore import Tensor
from numpy import cov
from numpy.random import random
from scipy.linalg import sqrtm
from src.configs import *
from data_loader import *
from GAN_model import *

set_seed(1)

reshape = mindspore.ops.Reshape()
log = mindspore.ops.Log()
exp = mindspore.ops.Exp()
cat = mindspore.ops.Concat()
cat2 = mindspore.ops.Concat(1)
squeeze1 = mindspore.ops.Squeeze(1)
sigma = None

#log_mean_exp函数计算平均值
def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = reshape(max_, (max_.shape[0], 1))
    return max_ + log(exp(a - max2).mean(1))

#使用parzen进行模型评估，设计窗口函数
def mind_parzen(x, mu, sigma):
    a = (reshape(x, (x.shape[0], 1, x.shape[-1])) - reshape(mu, (1, mu.shape[0], mu.shape[-1]))) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z

#通过get_nll函数获得似然值，其中的操作是每次输入一个batch_size,最后对各个batch_size求取平均值
def get_nll(x, samples, sigma, batch_size=10):
    inds = range(x.shape[0])
    inds = list(inds)
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    times = []
    nlls = Tensor(np.array([]).astype(np.float32))
    for i in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = cat((nlls, nll))
    return nlls

#在验证集上进行交叉检验
def cross_validate_sigma(samples, data, sigmas, batch_size):
    lls = Tensor(np.array([]).astype(np.float32))
    for sigma in sigmas:
        print("sigma=", sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = tmp.mean()
        tmp = reshape(tmp, (1, 1))
        tmp = squeeze1(tmp)
        lls = cat((lls, tmp))
        gc.collect()
    ind = lls.argmax()
    return sigmas[ind]

#获取有效的测试图像
def get_valid(limit_size=-1, fold=0):
    dataset1 = create_dataset_valid(batch_size=1000, repeat_size=1, latent_size=100)
    data = []
    for image in enumerate(dataset1):
        data = image
        break
    image = data[1]
    image = image[0]
    image = reshape(image, (image.shape[0], 784))
    return image

def get_test(limit_size):
    dataset_ = decode_idx3_ubyte(test_images_file).astype("float32")
    image = Tensor(dataset_)
    image = reshape(image, (image.shape[0], 784))
    return image[0:limit_size]

#使用对数最大似然进行评估
def parzen(samples):
    ll = [1]
    se = 1
    shape = samples.shape
    samples = reshape(samples, (shape[0], -1))
    sigma = None
    if sigma is None:
        valid = get_valid(limit_size=1000, fold=0)
        valid = valid.asnumpy()
        valid = valid / 255
        valid = Tensor(valid)

        sigma_range = np.logspace(-1, 0, num=5)
        sigma = cross_validate_sigma(samples, valid, sigma_range, batch_size_test)
    else:
        sigma = float(sigma)

    print("Using Sigma: {}".format(sigma))
    gc.collect()
    test_data = get_test(limit_size=1000)
    test_data = test_data / 255
    ll = get_nll(test_data, samples, sigma, batch_size=batch_size_test)
    se = ll.std() / np.sqrt(test_data.shape[0])
    return ll.mean(), se


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
test_noise = Tensor(np.random.normal(size=(1000, latent_size)), dtype=mstype.float32)
llh, x = [], []
def eval(n):
    model = Generator(latent_size=100)
    param_dict = mindspore.load_checkpoint("./result/checkpoints/Generator"+str(n)+".ckpt")
    mindspore.load_param_into_net(model, param_dict)
    # 生成数据
    samples1 = model(test_noise)
    fake_img = Tensor((samples1.asnumpy() * 127.5 + 127.5) / 255)
    fake_img = reshape(fake_img, (fake_img.shape[0], 784))
    mean_ll, se_ll = parzen(fake_img)
    x.append(n)
    llh.append(mean_ll)
    print("n = , Log-Likelihood of test set = {}, se: {}".format(n, mean_ll, se_ll))

for i in range(0, 100, 5):
    eval(i)
llh = [i.asnumpy() for i in llh]
plt.plot(x, llh)
eval_path = "./result/eval.png"
plt.savefig(eval_path)
plt.show()
