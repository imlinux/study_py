import os
import pathlib
from shutil import copyfile
import ppdet

with open("/home/dong/tmp/dataset/voc/no-PascalVOC-export/ImageSets/Main/0_train.txt") as f:
    for line in f:
        filename, t = line.replace("\n", "").split(" ")
        if t == "1":
            copyfile("/home/dong/tmp/dataset/voc/no-PascalVOC-export/JPEGImages/" + filename, "/home/dong/tmp/tmp/" + filename)