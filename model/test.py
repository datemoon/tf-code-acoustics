import sys
import tensorflow as tf
import numpy as np
sys.path.append("../")
from model.nnet_compoment import *

feat=np.arange(20).reshape(4,5)

norm=NormalizeLayer({})
sess=tf.Session()

sess.run(norm(feat))
'''
0           0.182574186 0.365148372 0.547722558 0.730296743
0.313112146 0.375734575 0.438357004 0.500979433 0.563601862
0.370116605 0.407128266 0.444139926 0.481151587 0.518163247
0.393241879 0.419458004 0.445674129 0.471890255 0.49810638
'''
feat3=feat.reshape(5,2,2)
a=tf.pad(feat3,[[2,0],[0,0],[0,0]],"SYMMETRIC")
b=tf.pad(feat3,[[1,1],[0,0],[0,0]],"SYMMETRIC")
c=tf.pad(feat3,[[0,2],[0,0],[0,0]],"SYMMETRIC")

concat_abc = tf.concat([a,b,c],-1)
sess.run(tf.concat([a,b,c],-1))

splice=SpliceLayer({})
#splice(b)
sess.run(splice(b))
sess.run(splice(b,False))

