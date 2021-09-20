import os
import pathlib

path = "/home/dong/tmp/no"
mode = "train"

with open(f'{path}/{mode}.txt', "w") as f:
    for i in pathlib.Path(path).glob("*/*"):
        *item, label, _ = str(i).split("/")
        print(f"{i} {label}", file=f)