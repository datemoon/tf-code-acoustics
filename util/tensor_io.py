from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, shutil, time
import random
import threading
import numpy as np
import time
import logging

import tensorflow as tf

def print_tensor(x, f=sys.stdout, end='\n', level=0, name=None):
    if len(x.shape) <= 0:
        return 
    if name:
        print(name + ' ' + str(x.shape) + ':', file=f)

    if len(x.shape) == 1:
        print('[ ', end='', file=f)
        for i in range(len(x) - 1):
            print(str(x[i]) + ' ', end='', file=f)
        print(str(x[-1]) + ']', end=end, file=f)
    else:
        print('[', end='', file=f)
        for d in range(x.shape[0] - 1):
            if d != 0:
                print(' '*(level+1), end='', file=f)
            print_tensor(x[d], f=f, end='\n', level=level+1)
        if len(x) != 1:
            print(' '*(level+1), end='', file=f)
        print_tensor(x[-1], f=f, end='', level=level+1)
        print(']', end='', file=f)
        if level > 0:
            print(end*(len(x[0].shape) + 1), end='', file=f)
        elif level == 0:
            print(end, end='', file=f)

def save_variables(variables, param, save_file):
    save_fp = open(save_file,'w')
    for key in range(len(variables)):
        print_tensor(param[key],name=variables[key].name, f=save_fp)
    save_fp.close()

def save_variables_1(param, save_file):
    save_fp = open(save_file,'w')
    for key in param.keys():
        print(key)
        print_tensor(param[key], name=key, f=save_fp)
    save_fp.close()

def print_trainable_variables(sess, save_file):
    variables=tf.trainable_variables()
    param = sess.run(variables)
    save_thread = threading.Thread(group=None, target=save_variables,
            args=(variables, param, save_file,),
            kwargs={}, name= 'save_trainable_variables')
    save_thread.start()


