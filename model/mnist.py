import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# define net
class caffe_mnist(nn.Module):
    def __init__(self, with_bn=False):
        super(caffe_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(800, 500)   # 4*4*50 = 800
        self.fc2 = nn.Linear(500, 10)
        self.weight_init()  # initialize weights ourselves
        self.with_bn = False
        if with_bn:
            self.with_bn = True
            self.conv1_bn = nn.BatchNorm2d(20)
            self.conv2_bn = nn.BatchNorm2d(50)
            self.fc1_bn = nn.BatchNorm1d(500)

    def weight_init(self):
        init.xavier_uniform(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        init.xavier_uniform(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        init.xavier_uniform(self.fc1.weight)
        init.constant(self.fc1.bias, 0)
        init.xavier_uniform(self.fc2.weight)
        init.constant(self.fc2.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        # print(x.shape)
        if self.with_bn:
            x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        # print(x.shape)
        if self.with_bn:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = x.view(-1, 800) # reshape
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        if self.with_bn:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)
