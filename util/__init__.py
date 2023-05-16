import json

import PIL.Image as Image
import matplotlib_inline
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def use_svg_display():
    # 用矢量图显示
    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, loss, epochs, device, optimizer):
    net = net.to(device)
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print(y_hat)
            # 正常情况下请使用这个
            l = loss(y_hat, y)
            # 出现block: [0,0,0], thread: [1,0,0] Assertion `t >= 0 && t < n_classes` failed.可尝试使用此行
            # l = loss(y_hat, y - 1)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))


def padding_fit(pil_img, bg_color, target_size):
    width, height = pil_img.size
    if width == height:
        return pil_img.resize((target_size, target_size))
    else:
        size = max(width, height)
        res = Image.new(pil_img.mode, (size, size), bg_color)
        res.paste(pil_img, (0, abs(width - height) // 2))
        return res.resize((target_size, target_size))


def center_cut_fit(pil_img, target_size):
    width, height = pil_img.size
    if width == height:
        return pil_img.resize((target_size, target_size), Image.ANTIALIAS)
    else:
        res = pil_img
        if width >= height:
            res = res.resize((int(width * (target_size / width)), target_size), Image.ANTIALIAS)
            box = ((width - target_size) // 2, 0, (width - target_size) // 2 + target_size, target_size)
            return res.crop(box)
        else:
            res = res.resize((target_size, int(height * (target_size / height))), Image.ANTIALIAS)
            box = (0, (height - target_size) // 2, target_size, (height - target_size) // 2 + target_size)
            return res.crop(box)


def resize_fit(pil_img, target_size):
    return pil_img.resize((target_size, target_size), Image.LANCZOS)


def test_model(net, X, y, fig_size, label_data=None, show_img=False):
    true_labels = y.numpy()
    pred = net(X)
    pred_labels = pred.argmax(dim=1).cpu().numpy()
    if label_data is None:
        titles = [str(true) + ',' + str(pred) for true, pred in zip(true_labels, pred_labels)]
    else:
        titles = [str(label_data[true]) + ',' + str(label_data[pred]) for true, pred in zip(true_labels, pred_labels)]
    print(titles)
    if show_img:
        use_svg_display()
        _, figs = plt.subplots(1, len(X), figsize=fig_size)
        for f, img, lbl in zip(figs, X, titles):
            f.imshow(transforms.ToPILImage()(img))
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()


def vgg_blk(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2)
    )


def cut_img(pil_img, left_x, width, height):
    box = (left_x, 0, left_x + width, height)
    return pil_img.crop(box)


def load_labels(file_path):
    with open(file_path) as label_file:
        label_dict = json.load(label_file)
    label_data = [i for i in range(len(label_dict))]
    for k, v in label_dict.items():
        label_data[v] = k
    return label_data


def load_valid_data(file_path):
    with open(file_path) as file:
        valid_file = file.readlines()
        valid = {}
        for line in valid_file:
            s = line.split(" ")
            valid[s[0]] = s[1]
    return valid
