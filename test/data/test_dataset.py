import paddle
import numpy as np
import cv2;

class MyDataset(paddle.io.Dataset):

    def __init__(self):
        super().__init__()
        self.data = []
        with open("/home/dong/tmp/mnist/train.txt") as f:
            for i in f:
                (img, label) = i.replace("\n", "").split(" ")
                img_data = cv2.imread(img, 0)
                self.data.append((img_data.astype("float32"), np.array(label).astype('float32')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


data_loader = paddle.io.DataLoader(MyDataset())

for i in data_loader():
    print(i)
    break

