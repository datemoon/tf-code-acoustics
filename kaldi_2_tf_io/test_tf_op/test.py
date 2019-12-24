import tensorflow as tf

import time


zero_out_module = tf.load_op_library('./test.so')
for i in range(100):
    start = time.time()
    res = zero_out_module.zero_out([5, 4, 3, 2, 1])
    end = time.time()
    print(i,res,'time:',end-start)


