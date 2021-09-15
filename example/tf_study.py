import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


img_raw = tf.io.read_file("/home/dong/tmp/mnist/1/train40.jpg")
#img_raw = tf.io.read_file("/home/dong/tmp/2.jpg")

img_tensor = tf.image.decode_image(img_raw, channels=1)
img_tensor = tf.image.resize(img_tensor, [28, 28])
#img_tensor = img_tensor / 255
img_tensor = tf.where(img_tensor < 100, 0, img_tensor)
print(img_tensor)

# img_tensor = tf.reshape(img_tensor, [1, 28, 28])
#
# print(img_tensor.shape);