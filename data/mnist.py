import os
import cv2

from paddle.vision.datasets.mnist import MNIST

mode = "train"
data_root_dir = "/home/dong/tmp/mnist"
data_set_dir = data_root_dir + "/" + mode

data = MNIST(mode=mode)
os.makedirs(data_set_dir, exist_ok=True)
with open(data_root_dir + "/" + mode + ".txt", mode="wt") as f:
    for i in range(len(data)):
        sample = data[i]
        img_dir = data_set_dir + "/" + str(sample[1][0])
        os.makedirs(img_dir, exist_ok=True)
        img_file = f'{img_dir}/{str(i)}.jpg'
        sample[0].save(img_file)
        print(f'{img_file} {str(sample[1][0])}', file=f)

        img = cv2.imread(img_file, 0)
        ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(img_file, img)