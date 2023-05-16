import glob
import torch
import torchvision
from PIL import Image
import util

model = torch.load("captcha-text.pt").to("cpu")
dir_path = r'E:\PycharmProject\Captcha\download\cut\*.*'
save_path = "E:\\PycharmProject\\Captcha\\train-data-text\\text-train\\"
prefix = "cut_"
res = glob.glob(dir_path)
cnt = 0
label_data = util.load_labels("E:\\PycharmProject\\Captcha\\train-data-text\\labelData.json")
for file in res:
    image = Image.open(file).convert("L")
    tmp_img = torchvision.transforms.ToTensor()(util.resize_fit(image, 32)).unsqueeze(0)
    pred = model(tmp_img).argmax(dim=1).numpy()[0]
    image.save(save_path + f"\\{label_data[pred]}\\{prefix}{str(cnt)}.jpg")
    cnt += 1
    print(f"{file}, pred: {label_data[pred]}, cnt: {cnt}")
