# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:57:08 2019

@author: peppazhang
"""

from tensorflow.python.platform import gfile
import tensorflow as tf
import os

pb_file_path = '/data/algceph/peppazhang/bin/douyin/newmodel/inceptionv4'

pb_file_name = 'inception_v4.pb'
sess = tf.Session()
with gfile.FastGFile(os.path.join(pb_file_path,pb_file_name),'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,name='')

sess.run(tf.global_variables_initializer())

shape = sess.graph.get_tensor_by_name('InceptionV4/Logits/PreLogitsFlatten/Reshape/shape')
print(shape)
    