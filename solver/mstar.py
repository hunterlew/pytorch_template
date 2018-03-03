import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# define optimizer
class mstar_solver():
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.config = config
        self.optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # define training and testing process
    def train(self, epoch):
        self.model.train()  # switch to training process
        for batch_idx, (data, target) in enumerate(self.data.train_loader):
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # step learning policy
            adjust_lr = self.config.lr * (0.1 ** (epoch // 10))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjust_lr
            self.optimizer.zero_grad()

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)
            # input()
            loss = F.nll_loss(output, target)  # the negative log likelihood loss, in average by default
            loss.backward()
            self.optimizer.step()  # update
            torch.save(self.model.state_dict(), 'save/mstar/net-epoch-' + str(epoch) + '.pkl')
            if batch_idx % self.config.display == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), self.data.__len__(),
                    100. * batch_idx / self.data.batch_size(), loss.data[0]))

    def test(self):
        self.model.eval()  # switch to evaluation process

        # evaluate on the train set
        train_loss = 0
        train_correct = 0
        for data, target in self.data.train_loader:
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)
            train_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        train_loss /= self.data.__len__()
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, train_correct, self.data.__len__(),
            100. * train_correct / self.data.__len__()))

        # evaluate on the test set
        test_loss = 0
        test_correct = 0
        for data, target in self.data.test_loader:
            if self.config.use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.config.gray:
                output = self.model(data[:, :1, :, :])
            else:
                output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= self.data.__len__(False)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, test_correct, self.data.__len__(False),
            100. * test_correct / self.data.__len__(False)))

        return train_loss, train_correct / self.data.__len__(), test_loss, test_correct / self.data.__len__(False)
