import tensorflow as tf
import numpy as np

a = np.array(np.arange(1, 1 + 24).reshape([1, 12, 2]), dtype=np.float32)

kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([1, 2, 2])
#kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])
conv1d = tf.nn.conv1d(a, kernel, 1, 'VALID')

b = np.array([np.arange(1,25,2),np.arange(2,25,2)],dtype=np.float32).reshape([1, 2, 12])
print(a)
print(b)
print(kernel)
conv1d_ncw = tf.nn.conv1d(b, kernel, 1, 'VALID',data_format='NCW')
maxpool_ncw = tf.layers.max_pooling1d(conv1d_ncw,[4],[4],padding='valid',data_format='channels_first',name='maxpool_ncw')
maxpool = tf.layers.max_pooling1d(conv1d,[4],[4],padding='valid',data_format='channels_last',name='maxpool')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    cnnres = sess.run(conv1d)
    maxpoolres = sess.run(maxpool)
    cnnres_ncw = sess.run(conv1d_ncw)
    maxpoolres_ncw = sess.run(maxpool_ncw)
    print(cnnres)
    print(np.shape(cnnres))
    print(maxpoolres)
    print(np.shape(maxpoolres))
    print(cnnres_ncw)
    print(np.shape(cnnres_ncw))
    print(maxpoolres_ncw)
    print(np.shape(maxpoolres_ncw))
