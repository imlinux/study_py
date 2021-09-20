import paddle

train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train', backend='cv2'), batch_size=5, shuffle=True)

for images, labels in train_loader():
    print(labels)
    break