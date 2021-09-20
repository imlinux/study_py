import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.vision.datasets import MNIST
from paddle.vision.transforms import ToTensor
import os


class MyDataset(paddle.io.Dataset):

    def __init__(self, mode = "train", transform = None):
        super().__init__()
        self.data = []
        self.transform = transform
        with open(f"/home/dong/tmp/mnist/{mode}.txt") as f:
            for i in f:
                (img, label) = i.replace("\n", "").split(" ")
                img_data = cv2.imread(img, 0)

                self.data.append((img_data, np.array([label]).astype('int64')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        image, label = self.data[item]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


# 定义训练过程
def train(model, opt, train_loader, valid_loader):
    # 开启0号GPU训练
    use_gpu = False
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start training ... ')
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 2000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            pred = F.softmax(logits)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')


# 创建模型
model = LeNet(num_classes=10)
# # 设置迭代轮数
EPOCH_NUM = 5

if os.path.exists("mnist.pdparams"):

    param_dict = paddle.load("mnist.pdparams")
    model.load_dict(param_dict)
else:
    # 设置优化器为Momentum，学习率为0.001
    opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
    # 定义数据读取器
    train_loader = paddle.io.DataLoader(MyDataset(mode='train', transform=ToTensor()), batch_size=10, shuffle=True)
    valid_loader = paddle.io.DataLoader(MyDataset(mode='test', transform=ToTensor()), batch_size=10)
    # 启动训练过程
    train(model, opt, train_loader, valid_loader)

model.eval()
img = cv2.imread("/home/dong/tmp/2021-09-20_15-46.png", 0)
img = cv2.resize(img, [28, 28])
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("", img)
cv2.waitKey(0)

img = ToTensor()(img)
img = img.reshape([1, 1, 28, 28])
pred = F.softmax(model(img))
print(pred)
print(paddle.argmax(pred))
