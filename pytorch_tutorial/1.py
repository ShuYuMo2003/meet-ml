import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DATASET_DIR = '../dataset'
BATCH_SIZE  = 64


# 加载需要的数据集
training_data = datasets.FashionMNIST(
    root=DATASET_DIR,
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root=DATASET_DIR,
    train=False,
    download=True,
    transform=ToTensor()
)

# 创建需要的 data loader
trai_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print(X.shape)
    print(y.shape)
    break