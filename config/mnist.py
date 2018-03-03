import torch


class mnist_config():
    def __init__(self):
        self.train_batch_size = 64
        self.test_batch_size = 100
        self.epochs = 10
        self.with_bn = False
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0001 if self.with_bn else 0.0005
        self.use_gpu = True and torch.cuda.is_available()
        self.seed = 0
        self.display = 100
