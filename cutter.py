import glob
import torch
import torchvision
from PIL import Image

import util

model = torch.load("captcha-bound.pt").to("cpu")
window_size = 16
img_width = 90
img_height = 30
dir_path = r'E:\PycharmProject\Captcha\download\*.*'
save_path = "E:\\PycharmProject\\Captcha\\download\\cut\\"
prefix = "cuta_"
res = glob.glob(dir_path)
cnt = 0
for file in res:
    img = Image.open(file).convert("L")
    i = 0
    while i <= img_width - window_size:
        image = util.cut_img(img, i, window_size, img_height)
        tmp_img = torchvision.transforms.ToTensor()(util.resize_fit(image, 32)).unsqueeze(0)
        pred = model(tmp_img).argmax(dim=1).numpy()[0]
        if pred == 1:
            i += window_size
            image.save(save_path + prefix + str(cnt) + ".jpg")
            cnt += 1
        else:
            i += 1
    print(cnt)
    # break
