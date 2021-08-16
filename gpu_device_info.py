# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:56:55 2020

@author: ferga
"""
import sys
import os
import tensorflow as tf


print(tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
