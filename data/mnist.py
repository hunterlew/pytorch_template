import torch
from torchvision import datasets, transforms


# load dataset
class mnist(torch.utils.data.Dataset):
    def __init__(self, train_bs, test_bs):
        super(mnist, self).__init__()
        # transformation
        self.transform = transforms.Compose([transforms.ToTensor()])  # 0~255 -> 0~1

        # return tuple (data, target)
        self.train_data = datasets.MNIST('download/mnist', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST('download/mnist', train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=train_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=test_bs, shuffle=True)

    def __len__(self, train=True):
        # return the number of training or testing samples
        return len(self.train_data) if train else len(self.test_data)

    def __getitem__(self, item, train=True):
        # return the tuple (data, target) by the index
        return self.train_data[item] if train else self.test_data[item]

    def size(self, train=True):
        # return the size of each training or testing sample
        return self.train_data[0][0].size() if train else self.test_data[0][0].size()

    def batch_size(self, train=True):
        # return the batch size
        return len(self.train_loader) if train else len(self.test_loader)
