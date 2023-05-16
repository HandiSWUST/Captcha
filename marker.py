import json
import os
import shutil

target_label = r"E:\PycharmProject\Captcha\train-data\labels.txt"
from_path = r"E:\PycharmProject\Captcha\train-data\success"
target_path = r"E:\PycharmProject\Captcha\train-data\train"
dir_list = os.listdir(from_path)
with open(target_label, "a") as labels:
    for file in dir_list:
        if "labels" not in file:
            labels.write(file + " 1\n")
            # os.system(f"xcopy {from_path}\\{file} {target_path}")
