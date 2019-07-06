# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:37:39 2019

@author: peppazhang
"""
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import functools


class dataset():
    
    def __init__(self,filedir, filename, split_ratio, train_batch_size, test_batch_size):
        
        self.filedir = filedir
        self.filename = filename
        self.split_ratio = split_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.sess = tf.Session()
        
        self.train_X, self.test_X, self.train_y, self.test_y = self.train_test()
        
    def _init_train_iter(self):
        
        
        types=(tf.string, tf.float32,tf.int32, tf.float32, tf.float32)

        self.dataset = tf.data.Dataset.from_generator(functools.partial(self.train_generator_fn,self.train_X, self.train_y), output_types = types)
        self.dataset = self.dataset.batch(self.train_batch_size)
        self.train_dataset = self.dataset.shuffle(buffer_size = 20* self.train_batch_size)
        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.sess.run(self.train_iter.initializer)
    
    def _init_test_iter(self):  
        
        types=(tf.string, tf.float32,tf.int32, tf.float32, tf.float32)
        self.dataset = tf.data.Dataset.from_generator(functools.partial(self.train_generator_fn,self.test_X, self.test_y), output_types = types)
        self.test_dataset = self.dataset.batch(self.test_batch_size)
        self.test_iter = self.test_dataset.make_initializable_iterator()
        self.sess.run(self.test_iter.initializer)
        
    def train_test(self):
        
        f = open(os.path.join(self.filedir,self.filename),'r')
        data = f.readlines()
        n = len(data)
        
        feat = []
        label = []
        
        for i in range(n):
            item = data[i].strip().split('\t')
            feat.append(item[:-1])
            label.append(item[-1])
            
        train_X, test_X, train_y, test_y = train_test_split(feat, label, test_size = 0.2, random_state = 0)
        
        return train_X, test_X, train_y, test_y

    def generator_fn(self, X, y):
        for feat, label in zip(X, y):
            try:
                yield self.parse_fn(feat,label)
            except:
                print("input error")
                continue
    
    def parse_fn(self,feat,label):
        
        return feat,label
    
    
    def _yield_batch_traindata(self):
        
        feat, label = self.train_iter.get_next()
        feat, label  = self.sess.run([feat,label])
        
        return feat, label
    
    def _yield_batch_testdata(self):
        
        feat, label = self.test_iter.get_next()
        feat, label  = self.sess.run([feat,label])
        
        return feat, label