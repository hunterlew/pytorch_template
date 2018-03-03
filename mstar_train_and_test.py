from __future__ import print_function
import torch
from data.mstar import mstar
from model.mstar import CNN_8, CNN_12
from solver.mstar import mstar_solver
from config.mstar import mstar_config
from utils.visualize import *


# loading configuration
config = mstar_config()
if config.use_gpu:
    torch.cuda.manual_seed(config.seed)
else:
    torch.manual_seed(config.seed)


# loading dataset
data = mstar(config.train_batch_size, config.test_batch_size)

# loading net
model = CNN_12(config.with_bn)
if config.use_gpu:
    model.cuda()

# loading solver
optim = mstar_solver(data, model, config)

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
    draw_curves(config.epochs, x, y1, y2, y3, y4, 'mstar_CNN-12_bn')  # draw the curves


