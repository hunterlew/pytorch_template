from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


# load dataset
class mstar():
    def __init__(self, train_bs, test_bs):
        self.transform = transforms.Compose([transforms.ToTensor()])    # 0~255 --> 0~1

        self.train_data = ImageFolder('download/mstar/train', self.transform)
        self.test_data = ImageFolder('download/mstar/test', self.transform)

        self.train_loader = DataLoader(self.train_data, train_bs, True)
        self.test_loader = DataLoader(self.test_data, test_bs, True)

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
