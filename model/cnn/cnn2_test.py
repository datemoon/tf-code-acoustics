import tensorflow as tf
import numpy as np

a = np.array(np.arange(1, 1 + 24).reshape([1, 12, 2, 1]), dtype=np.float32)
data_format = "NHWC"
strides = [1,1,1,1]
filters = [1, 2, 1, 2]

kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape(filters)
#kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])
conv2d = tf.nn.conv2d(a, kernel, strides = [1,1,1,1], padding = 'VALID')

print(a)
print(kernel)
maxpool = tf.layers.max_pooling2d(conv2d,[4,1],[4,1],padding='valid',data_format='channels_last',name='maxpool')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    cnnres = sess.run(conv2d)
    maxpoolres = sess.run(maxpool)
    print(cnnres)
    print(np.shape(cnnres))
    print(maxpoolres)
    print(np.shape(maxpoolres))
