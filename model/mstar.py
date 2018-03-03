import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# define net
class CNN_8(nn.Module):
    def __init__(self, with_bn=False):
        super(CNN_8, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)
        self.weight_init()  # initialize weights ourselves
        self.with_bn = False
        if with_bn:
            self.with_bn = True
            self.conv1_bn = nn.BatchNorm2d(16)
            self.conv2_bn = nn.BatchNorm2d(32)
            self.conv3_bn = nn.BatchNorm2d(128)
            self.fc1_bn = nn.BatchNorm1d(128)

    def weight_init(self):
        init.xavier_uniform(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        init.xavier_uniform(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        init.xavier_uniform(self.conv3.weight)
        init.constant(self.conv3.bias, 0)
        init.xavier_uniform(self.fc1.weight)
        init.constant(self.fc1.bias, 0)
        init.xavier_uniform(self.fc2.weight)
        init.constant(self.fc2.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 6)
        # print(x.shape)
        if self.with_bn:
            x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 4)
        # print(x.shape)
        if self.with_bn:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x)
        # print(x.shape)
        if self.with_bn:
            x = self.conv3_bn(x)
        x = F.relu(x)
        x = x.view(-1, 128)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        # print(x.shape)
        if self.with_bn:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class CNN_12(nn.Module):
    def __init__(self, with_bn=False):
        super(CNN_12, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)
        self.weight_init()  # initialize weights ourselves
        self.with_bn = False
        if with_bn:
            self.with_bn = True
            self.conv1_bn = nn.BatchNorm2d(16)
            self.conv2_bn = nn.BatchNorm2d(32)
            self.conv3_bn = nn.BatchNorm2d(64)
            self.conv4_bn = nn.BatchNorm2d(128)
            self.conv5_bn = nn.BatchNorm2d(256)
            self.fc1_bn = nn.BatchNorm1d(256)

    def weight_init(self):
        init.xavier_uniform(self.conv1.weight)
        init.constant(self.conv1.bias, 0)
        init.xavier_uniform(self.conv2.weight)
        init.constant(self.conv2.bias, 0)
        init.xavier_uniform(self.conv3.weight)
        init.constant(self.conv3.bias, 0)
        init.xavier_uniform(self.conv4.weight)
        init.constant(self.conv4.bias, 0)
        init.xavier_uniform(self.conv5.weight)
        init.constant(self.conv5.bias, 0)
        init.xavier_uniform(self.fc1.weight)
        init.constant(self.fc1.bias, 0)
        init.xavier_uniform(self.fc2.weight)
        init.constant(self.fc2.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        if self.with_bn:
            x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        if self.with_bn:
            x = self.conv2_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv3(x), 3, stride=2)
        if self.with_bn:
            x = self.conv3_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv4(x), 2)
        if self.with_bn:
            x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.with_bn:
            x = self.conv5_bn(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        if self.with_bn:
            x = self.fc1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
