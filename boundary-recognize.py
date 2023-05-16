import torch.nn as nn
import torch.optim
import torch.utils.data as Data
import torchvision.transforms as transforms

import util
from util.MyDataset import MyDataset

image_size = 32
batch_size = 768
num_inputs = int((image_size / 32) ** 2 * 512)  # w * h * c
num_outputs = 4
epochs = 30
device = torch.device("cuda:0")
train_dataset = MyDataset("E:\\PycharmProject\\Captcha\\train-data\\labels.txt",
                          "E:\\PycharmProject\\Captcha\\train-data\\train", image_size,
                          transform=transforms.ToTensor(), label_bias=0, color_mode="L")
test_dataset = MyDataset("E:\\PycharmProject\\Captcha\\train-data\\valid.txt",
                         "E:\\PycharmProject\\Captcha\\train-data\\test", image_size, transform=transforms.ToTensor(),
                         label_bias=0, color_mode="L")
train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
net = nn.Sequential(
    util.vgg_blk(1, 16),
    util.vgg_blk(16, 32),
    util.FlattenLayer(),
    nn.Linear(int(32 * (image_size / 4) * (image_size / 4)), 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()
util.train(net, train_iter, test_iter, loss, epochs, device, optimizer)

# 保存模型
torch.save(net, 'captcha-bound.pt')

# 测试
net = torch.load("captcha-bound.pt")
X, y = iter(test_iter).__next__()
util.test_model(net, X.cuda(), y, (32, 32))
