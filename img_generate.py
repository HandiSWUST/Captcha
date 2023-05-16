import glob

from PIL import Image


def open_img(img_path):
    image = Image.open(img_path).convert("L")
    return image


def cut_img(pil_img, left_x, width, height):
    box = (left_x, 0, left_x + width, height)
    return pil_img.crop(box)


window_size = 16
img_width = 90
img_height = 30
bias = 616
cnt = 0
with open("E:\\PycharmProject\\Captcha\\captcha-data\\cut\\labels.txt", "a") as labels:
    dir_path = r'E:\PycharmProject\Captcha\captcha-data\*.*'
    res = glob.glob(dir_path)
    for file in res:
        img = open_img(file)
        for i in range(0, img_width, 2):
            img_cut = cut_img(img, i, window_size, img_height)
            img_cut.save(".\\captcha-data\\cut\\cap_" + str(cnt + bias) + ".jpg")
            labels.write("cap_" + str(cnt + bias) + ".jpg" + "\n")
            cnt += 2
