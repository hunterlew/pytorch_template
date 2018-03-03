from __future__ import print_function
import torch
from data.mnist import mnist
from model.mnist import caffe_mnist
from solver.mnist import mnist_solver
from config.mnist import mnist_config
from utils.visualize import *


# loading configuration
config = mnist_config()
if config.use_gpu:
    torch.cuda.manual_seed(config.seed)
else:
    torch.manual_seed(config.seed)


# loading dataset
data = mnist(config.train_batch_size, config.test_batch_size)

# loading net
model = caffe_mnist(config.with_bn)
if config.use_gpu:
    model.cuda()

# loading solver
optim = mnist_solver(data, model, config)

# main process
x = []
y1 = []
y2 = []
y3 = []
y4 = []
for epoch in range(1, config.epochs + 1):
    optim.train(epoch)
    train_loss, train_accuracy, test_loss, test_accuracy = optim.test()
    x += [epoch]
    y1 += [train_loss]
    y2 += [test_loss]
    y3 += [train_accuracy]
    y4 += [test_accuracy]
    draw_curves(config.epochs, x, y1, y2, y3, y4, 'test')  # draw the curves


