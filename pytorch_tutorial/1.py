import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

_dataset_dir = '../dataset'
_batch_size  = 64
_device = ("cuda" if torch.cuda.is_available() else "cpu")

print('using', _device)

# 加载需要的数据集
training_data = datasets.FashionMNIST(
    root=_dataset_dir,
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root=_dataset_dir,
    train=False,
    download=True,
    transform=ToTensor()
)

# 创建需要的 data loader
trai_dataloader = DataLoader(training_data, batch_size=_batch_size)
test_dataloader = DataLoader(test_data, batch_size=_batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_rule_stack(x)
        return logits


model = NeuralNetwork().to(_device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y)in enumerate(dataloader):
        X, y = X.to(_device), y.to(_device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"batch={batch}, loss={loss}, ({(batch+1) * X.shape[0]}/{size})")

def test(dataloader, model, loss_fn, optimizer):
    accruacy = 0
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(_device), y.to(_device)
            pred = model(X)
            accruacy += (y == pred.argmax(1)).type(torch.float).sum().item()
            loss += loss_fn(pred, y).item()


    loss /= len(dataloader)
    accruacy /= len(dataloader.dataset)

    print(f"loss={loss}, accruacy={accruacy}")


for epoch in range(5):
    print(f'epoch={epoch}')
    train(trai_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, optimizer)
