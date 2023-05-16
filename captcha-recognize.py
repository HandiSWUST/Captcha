import glob

import torch
import torchvision.transforms
from PIL import Image, ImageEnhance

import util


def open_img(img_path):
    image = Image.open(img_path).convert("L")
    return image


model_boundary = torch.load("captcha-bound.pt")
model_text = torch.load("captcha-text.pt")
window_size = 16
img_width = 90
img_height = 30
label_data = util.load_labels("E:\\PycharmProject\\Captcha\\train-data-text\\labelData.json")
dir_path = r'E:\PycharmProject\Captcha\captcha-data\test\*.*'
res = glob.glob(dir_path)
valid = util.load_valid_data(r"E:\PycharmProject\Captcha\captcha-data\test.txt")
total_acc = 0
for i in range(10):
    success, fail = 0, 0
    for file in res:
        captcha_str = ""
        img = open_img(file)
        i = 0
        while i <= img_width - window_size:
            image = util.cut_img(img, i, window_size, img_height)
            enh_col = ImageEnhance.Contrast(image)
            contrast = 10
            image = enh_col.enhance(contrast)
            tmp_img = torchvision.transforms.ToTensor()(util.resize_fit(image, 32)).unsqueeze(0).cuda()
            pred = model_boundary(tmp_img).argmax(dim=1).cpu().numpy()[0]
            if pred == 1:
                word_index = model_text(tmp_img).argmax(dim=1).cpu().numpy()[0]
                captcha_str += label_data[word_index]
                i += window_size
                # image.show()
            else:
                i += 2
        if valid[file].strip() == captcha_str:
            success += 1
        else:
            fail += 1
            # print(f"{file} {captcha_str} {valid[file].strip()}")
    total_acc += success / (success + fail)
    print(f"acc: {success / (success + fail)}")
print(f"total acc: {total_acc / 10}")

