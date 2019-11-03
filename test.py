import numpy as np
import keras.backend as K
from keras.layers import Conv2DTranspose
import tensorflow as tf

# i = np.random.random((1, 16, 16, 512))
# i = K.variable(i)
# # (1, 6, 6, 3)
# for j in range(3):
#     i = Conv2DTranspose(256, 4, strides=2, padding='same')(i)
# print(i.shape)

a = tf.reshape(tf.range(24), (2, 12))
i = tf.constant([[3], [4]])
sess = tf.Session()
print(sess.run(tf.batch_gather(a, i)))
b = tf.reshape(tf.range(24), (2, 6, 2))
print(sess.run(tf.batch_gather(b, i)))
