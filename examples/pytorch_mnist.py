import time
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from yaml import load


class Ds(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Metrics:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)
        self.correct += (pred == target).sum().item()
        self.total += len(target)
        return self.correct / self.total

    def reset(self):
        self.__init__()


def readLabel(label_dir: str):
    # 读取标签数据
    with open(label_dir, 'rb') as f:
        return np.fromfile(f, offset=8, dtype="int8")


def readImages(image_dir: str):
    # 读取图片数据
    with open(image_dir, 'rb') as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        num = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        columns = int.from_bytes(f.read(4), byteorder="big")
        return np.fromfile(f, dtype="uint8").reshape([num, rows, columns])


def get_loader(batch_size: int, shuffle: bool = True):
    # 读取数据
    image_dir = "E:\\ProjectFiles\\Python\\04_DeepLearning\\Datasets\\mnist\\raw\\train-images-idx3-ubyte"
    label_dir = "E:\\ProjectFiles\\Python\\04_DeepLearning\\Datasets\\mnist\\raw\\train-labels-idx1-ubyte"
    images = readImages(image_dir)
    labels = readLabel(label_dir)
    images = images.reshape([images.shape[0], -1]) / 255.0
    # 创建数据集
    ds = Ds(images, labels)
    # 创建数据加载器
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    model = Model()
    loader = get_loader(25)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    accu = Metrics()
    for i in range(10):
        now = time.time()
        for batch, (x, y) in enumerate(loader, 1):
            pred = model(x)
            loss = loss_fn(pred, y)
            if batch % 100 == 99 or batch == 0:
                print(f"[{i + 1}/10]batch: {batch + 1}/{len(loader)}, "
                      f"loss {loss.item():.6f}, "
                      f"acc {accu.update(pred, y):.4f} "
                      f"{(time.time() - now) / (batch + 1) * 1000:.2}ms/batch",
                      end="    \r")
            optim.zero_grad()
            loss.backward()
            optim.step()
        print()
