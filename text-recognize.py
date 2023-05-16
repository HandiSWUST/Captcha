import json

import torch.nn as nn
import torch.optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import util
from util.MyDataset import MyDataset

image_size = 32
batch_size = 512
num_inputs = int((image_size / 32) ** 2 * 512)  # w * h * c
num_outputs = 4
epochs = 10
device = torch.device("cuda:0")
train_dataset = MyDataset("E:\\PycharmProject\\Captcha\\train-data-text\\labels.txt",
                          "E:\\PycharmProject\\Captcha\\train-data-text\\text-train", image_size,
                          transform=transforms.ToTensor(), label_bias=0, color_mode="L")
test_dataset = MyDataset("E:\\PycharmProject\\Captcha\\train-data-text\\valid.txt",
                         "E:\\PycharmProject\\Captcha\\train-data-text\\text-test", image_size, transform=transforms.ToTensor(),
                         label_bias=0, color_mode="L")
train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
net = nn.Sequential(
    util.vgg_blk(1, 16),
    nn.BatchNorm2d(16),
    util.vgg_blk(16, 32),
    nn.BatchNorm2d(32),
    util.vgg_blk(32, 64),
    nn.BatchNorm2d(64),
    util.FlattenLayer(),
    nn.Linear(int(64 * (image_size / 8) * (image_size / 8)), 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 36)
)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
loss = nn.CrossEntropyLoss()
util.train(net, train_iter, test_iter, loss, epochs, device, optimizer)

# 保存模型
torch.save(net, 'captcha-text.pt')

# 测试
with open("E:\\PycharmProject\\Captcha\\train-data-text\\labelData.json") as label_file:
    label_dict = json.load(label_file)
label_data = [i for i in range(len(label_dict))]
for k, v in label_dict.items():
    label_data[v] = k
net = torch.load("captcha-text.pt")
X, y = iter(test_iter).__next__()
util.test_model(net, X.cuda(), y, (32, 32), label_data=label_data)
