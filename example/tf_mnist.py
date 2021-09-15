import os

import tensorflow as tf
import matplotlib.pyplot as plt

model_file = "/home/dong/tmp/mnist_model"

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if not os.path.exists(model_file):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.save(model_file);

else:
    model = tf.keras.models.load_model(model_file)

#img_raw = tf.io.read_file("/home/dong/tmp/8.png")
#img_raw = tf.io.read_file("/home/dong/Downloads/8_preview_rev_1.png")
# img_raw = tf.io.read_file("/home/dong/tmp/mnist/6/test11.jpg")
img_raw = tf.io.read_file("/home/dong/Downloads/2021-09-14_18-22_preview_rev_1.png")

img_tensor = tf.image.decode_image(img_raw, channels=1)

# img_tensor = 255 - img_tensor
# img_tensor = tf.where(img_tensor >= 150, img_tensor, 0)
#
# t = tf.image.encode_jpeg(img_tensor, quality=100, format='grayscale')
# tf.io.write_file('/home/dong/tmp/test_sinogram.jpg', t)

#img_tensor = tf.where(img_tensor != 0, 255 - img_tensor, 0)
img_tensor = img_tensor / 255
img_tensor = tf.image.resize(img_tensor, [28, 28])
img_tensor = tf.reshape(img_tensor, [1, 28, 28])
print(img_tensor)

plt.figure(1)
plt.imshow(img_tensor[0], cmap='gray')
plt.show()

print(tf.argmax(model.predict(img_tensor)[0]))
print(model.predict(img_tensor))