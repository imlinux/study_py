import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt


def process(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])

    image = image / 255
    image = tf.reshape(image, [28, 28])

    return image, label

def process1(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)

    _, image = cv2.threshold(image.numpy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    image = tf.image.resize(image, [28, 28])

    image = image / 255
    image = tf.reshape(image, [28, 28])

    return image, label


def ds():
    path_list = []
    label_list = []
    with open("/home/dong/tmp/mnist/train.txt") as f:
        for i in f:
            path, label = i.replace("\n", "").split(" ")
            path_list.append(path)
            label_list.append(int(label))

    ret = tf.data.Dataset.from_tensor_slices((path_list, label_list))
    ret = ret.map(process)
    ret = ret.batch(5)

    return ret


def main():
    model_file = "/home/dong/tmp/mnist_model_1"

    if not os.path.exists(model_file):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(ds(), epochs=5)

        model.save(model_file)

    else:
        model = tf.keras.models.load_model(model_file)

    img_tensor, _ = process1('/home/dong/tmp/2021-09-20_01-27.png', 1)
    img_tensor = tf.reshape(img_tensor, [1, 28, 28])

    plt.figure(1)
    plt.imshow(img_tensor[0], cmap='gray')
    plt.show()

    print(tf.argmax(model.predict(img_tensor)[0]))
    print(model.predict(img_tensor))


if __name__ == "__main__":
    main()
