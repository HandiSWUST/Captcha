import json
import os
import shutil

target_label = r"E:\PycharmProject\Captcha\train-data-text\text-train\labels.txt"
base_path = r"E:\PycharmProject\Captcha\train-data-text\text-train"

with open("E:\\PycharmProject\\Captcha\\train-data-text\\labelData.json") as label_file:
    label_dict = json.load(label_file)
label_data = [i for i in range(len(label_dict))]
for k, v in label_dict.items():
    label_data[v] = k

for index, item in enumerate(label_data):
    dir_path = base_path + "\\" + str(item)
    dir_list = os.listdir(dir_path)
    with open(target_label, "a") as labels:
        for file in dir_list:
            if "labels" not in file:
                labels.write(file + " " + str(index) + "\n")
                os.popen(f"xcopy {dir_path}\\{file} {base_path}")
