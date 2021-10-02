import cv2
import pathlib
import os


def split(img_path, output, mode = 1):

    *_, filename = img_path.split("/")
    img = cv2.imread(img_path)

    os.makedirs(output, exist_ok = True)

    h = 106

    if mode == 1:
        columns = [(994, 1160), (1160, 1316), (1472, 1644), (1644,1800), (1942, 2108), (2108, 2270)]
        first_row = 670
        row_cnt = 24
    else:
        columns = [(994, 1160), (1160, 1316), (1472, 1644), (1644,1800), (1942, 2108), (2108, 2270)]
        first_row = 408
        row_cnt = 25

    for i in range(row_cnt):
        row = first_row + i * h
        for j, column in enumerate(columns):
            cv2.imwrite(f"{output}/{filename}-{i}-{j}.jpg", img[row: row + h, column[0]: column[1]])


for f in pathlib.Path("/home/dong/tmp/score/img/0").iterdir():
    split(str(f), "/home/dong/tmp/dataset/voc/no", 0)

for f in pathlib.Path("/home/dong/tmp/score/img/1").iterdir():
    split(str(f), "/home/dong/tmp/dataset/voc/no", 1)
