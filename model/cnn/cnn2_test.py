import tensorflow as tf
import numpy as np

a = np.array(np.arange(1, 1 + 24).reshape([1, 12, 2, 1]), dtype=np.float32)
b = np.array([np.arange(1,25,2),np.arange(2,25,2)],dtype=np.float32).reshape([1, 2, 12, 1])
data_format = "NHWC"
strides = [1,1,1,1]
afilters = [1, 2, 1, 2]
bfilters = [2, 1, 1, 2]

akernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape(afilters)
bkernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape(bfilters)
#kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])
aconv2d = tf.nn.conv2d(a, akernel, strides = [1,1,1,1], padding = 'VALID')
amaxpool = tf.layers.max_pooling2d(aconv2d,[4,1],[4,1],padding='valid',data_format='channels_last',name='maxpool')
bconv2d = tf.nn.conv2d(b, bkernel, strides = [1,1,1,1], padding = 'VALID')
bmaxpool = tf.layers.max_pooling2d(bconv2d,[1,4],[1,4],padding='valid',data_format='channels_last',name='maxpool')

print(a)
print(akernel)
print(b)
print(bkernel)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    acnnres = sess.run(aconv2d)
    amaxpoolres = sess.run(amaxpool)
    print(acnnres)
    print(np.shape(acnnres))
    print(amaxpoolres)
    print(np.shape(amaxpoolres))
    bcnnres = sess.run(bconv2d)
    bmaxpoolres = sess.run(bmaxpool)
    print(bcnnres)
    print(np.shape(bcnnres))
    print(bmaxpoolres)
    print(np.shape(bmaxpoolres))
