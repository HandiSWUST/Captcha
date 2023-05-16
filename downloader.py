import requests

prefix = "captcha_"
cnt = 0
url = "http://cas.swust.edu.cn/authserver/captcha"
num = 1500
path = "E:\\PycharmProject\\Captcha\\download\\"
for i in range(num):
    captcha_file = requests.get(url)
    open(path + prefix + str(cnt) + ".jpeg", "wb").write(captcha_file.content)
    print(str(cnt + 1) + "/" + str(num))
    cnt += 1

