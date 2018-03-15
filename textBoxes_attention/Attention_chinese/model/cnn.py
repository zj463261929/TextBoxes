#coding=utf-8
__author__ = 'moonkey'

#from keras import models, layers
import logging
import numpy as np
# from src.data_util.synth_prepare import SynthGen

#import keras.backend as K
import tensorflow as tf


def var_random(name, shape, regularizable=False):
    '''
    Initialize a random variable using xavier initialization.
    Add regularization if regularizable=True
    :param name:
    :param shape:
    :param regularizable:
    :return:
    '''
    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()) #通过所给的名字创建或是返回一个变量.
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'): #name_scope只能管住操作ops的名字，x.op.name
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v)) #将正则化方法应用到loss参数上， output = sum(v ** 2) / 2
    return v

def max_2x2pool(incoming, name):
    '''
    max pooling on 2 dims.
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID') #ksize=(1, 2, 2, 1)(batch_size,h,w,c)

def max_2x1pool(incoming, name):
    '''
    max pooling only on image width
    :param incoming:
    :param name:
    :return:
    '''
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='VALID')

def ConvRelu(incoming, num_filters, filter_size, name):
    '''
    Add a convolution layer followed by a Relu layer.
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :return:
    '''
    #a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组（tuple）(batch_size,h,w,c)
    num_filters_from = incoming.get_shape().as_list()[3]#可以使用 as_list()得到具体的尺寸[batch_size,h,w,c]
    with tf.variable_scope(name):#scope-范围， 变量的全称将会由当前变量作用域名+所提供的名字所组成，如v.name == "conv1/w"
        conv_W = var_random('W', tuple(filter_size) + (num_filters_from, num_filters), regularizable=True) 
        #(filter_size.get_shape().as_list[0], filter_size.get_shape().as_list[1], num_filters_from, num_filters)
        #conv1:(3,3,1,64)

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1,1,1,1), padding='SAME') #same的含义是：长度除以步长向上取整。out_height = ceil(float(in_height) / float(strides[1]))

        return tf.nn.relu(after_conv)#卷积层后接激活函数


def batch_norm(incoming, is_training):
    '''
    batch normalization
    :param incoming:
    :param is_training:
    :return:
    '''
    return tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, decay=0.99) #批处理归一化


def ConvReluBN(incoming, num_filters, filter_size, name, is_training, padding_type = 'SAME'):
    '''
    Convolution -> Batch normalization -> Relu
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :param is_training:
    :param padding_type:
    :return:
    '''
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random('W', tuple(filter_size) + (num_filters_from, num_filters), regularizable=True)

        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1,1,1,1), padding=padding_type)

        after_bn = batch_norm(after_conv, is_training)#

        return tf.nn.relu(after_bn)

def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)

def tf_create_attention_map(incoming):
    '''
    flatten hight and width into one dimention of size attn_length
    :param incoming: 3D Tensor [batch_size x cur_h x cur_w x num_channels]
    :return: attention_map: 3D Tensor [batch_size x attn_length x attn_size].
    '''
    shape = incoming.get_shape().as_list()
    print("shape of incoming is: {}".format(incoming.get_shape()))
    print(shape)
    return tf.reshape(incoming, (-1, np.prod(shape[1:3]), shape[3]))

class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()

    """

    def __init__(self, input_tensor, is_training):
        self._build_network(input_tensor, is_training)

    def _build_network(self, input_tensor, is_training):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        print('input_tensor dim: {}'.format(input_tensor.get_shape())) #(?, 1, 32, ?) #(batch_size,c,h,w)
        net = tf.transpose(input_tensor, perm=[0, 2, 3, 1]) #(batch_size,h,w,c)
        net = tf.add(net, (-128.0))
        net = tf.multiply(net, (1/128.0))#将输入网络的数据归一化到[-1,1]

        net = ConvRelu(net, 64, (3, 3), 'conv_conv1')
        net = max_2x2pool(net, 'conv_pool1')
        print("Layer 1: " + str(net.get_shape())) #(?, h,w,64)

        net = ConvRelu(net, 128, (3, 3), 'conv_conv2')
        net = max_2x2pool(net, 'conv_pool2')
        print("Layer 2: " + str(net.get_shape())) #(?, new_h,new_w,128)

        net = ConvReluBN(net, 256, (3, 3), 'conv_conv3', is_training)
        net = ConvRelu(net, 256, (3, 3), 'conv_conv4')
        net = max_2x1pool(net, 'conv_pool3')
        print("Layer 3: " + str(net.get_shape())) #(?, new_h,new_w,128)
		
        net = ConvReluBN(net, 512, (3, 3), 'conv_conv5', is_training)
        net = ConvRelu(net, 512, (3, 3), 'conv_conv6')
        net = max_2x1pool(net, 'conv_pool4')
        print("Layer 4: " + str(net.get_shape())) #(?, new_h,new_w,128)

        net = ConvReluBN(net, 512, (2, 2), 'conv_conv7', is_training, "VALID") 
        net = dropout(net, is_training) #Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作    
        print("Layer 5: " + str(net.get_shape())) #(?, new_h,new_w,128)
        print('CNN outdim before squeeze: {}'.format(net.get_shape()))  # 1x32x100 -> 24x512
        net = tf.squeeze(net,axis=1) #挤压， 删除axis=1维度， （batch_size, 24, c）

        print('CNN outdim: {}'.format(net.get_shape()))
        self.model = net

    def tf_output(self):
        # if self.input_tensor is not None:
        return self.model#, self.model1
    '''
    def __call__(self, input_tensor):
        return self.model(input_tensor)
    '''
    def save(self):
        pass


