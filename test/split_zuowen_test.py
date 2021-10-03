import cv2
import pathlib
import os


def split(img_path, output):

    *_, filename = img_path.split("/")
    img = cv2.imread(img_path)

    os.makedirs(output, exist_ok = True)

    row = 790
    col = 1716
    w = 100
    h = 76

    for i in range(6):
        cv2.imwrite(f"{output}/{filename}-{row}-{i}.jpg", img[row: row + h, col + i * w: col + (i + 1) * w])

    row = 3230
    col = 232
    w = 414 - col
    h = 3351 - row

    for i in range(8):
        cv2.imwrite(f"{output}/{filename}-{row}-{i}.jpg", img[row: row + h, col + i * w: col + (i + 1) * w])


# split("/home/dong/tmp/zuowen/img/0/JUYE_F_00007.pdf-17.jpg", "/home/dong/tmp/dataset/voc/zuowen")
for f in pathlib.Path("/home/dong/tmp/zuowen/img/0").iterdir():
    split(str(f), "/home/dong/tmp/dataset/voc/zuowen")
