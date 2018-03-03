import torch


class mstar_config():
    def __init__(self):
        self.train_batch_size = 32
        self.test_batch_size = 32
        self.gray = True
        self.with_bn = True
        self.epochs = 15
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0001 if self.with_bn else 0.0005
        self.use_gpu = True and torch.cuda.is_available()
        self.seed = 0
        self.display = 20
