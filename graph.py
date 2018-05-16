# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:08:24 2018

@author: Kel
"""
import tensorflow as tf

def deconv2d(x, W):
    """inverse convolution layer"""
    s = tf.multiply(tf.shape(x)[:3], [1,2,2])
    s = tf.stack([s[0], s[1], s[2], tf.shape(W)[2]])
    return tf.nn.conv2d_transpose(x, W, s, [1, 2, 2, 1])
    
def deconv3d(x, W, s):
    """inverse convolution layer"""
    shape_a = tf.multiply(tf.shape(x)[:4], [1,s,s,s])
    shape = tf.concat([shape_a, [tf.shape(W)[3]]], 0)
    return tf.nn.conv3d_transpose(x, W, shape, [1, s, s, s, 1])
    
def conv2d(x, W, s):
    """conv2d returns a 2d convolution layer with stride s."""
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

def conv3d(x, W, s):
    """conv3d returns a 3d convolution layer with stride s."""
    return tf.nn.conv3d(x, W, strides=[1, s, s, s, 1], padding='SAME')

def conv2d_blk(x, shape, stride):
    """conv2d block"""
    W = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=shape[3], initializer=tf.constant_initializer(0.1))
    return conv2d(x, W, stride) + b

def conv2d_relu(x, shape, stride):
    """conv2d block with ReLu"""
    W = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=shape[3], initializer=tf.constant_initializer(0.1))
    return conv2d(tf.nn.relu(x), W, stride) + b
    
def conv3d_blk(x, shape, stride, phase):
    """conv3d block with ReLu"""
    W = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=shape[4], initializer=tf.constant_initializer(0.1))
    return conv3d(tf.nn.relu(tf.contrib.layers.batch_norm(x, is_training=phase)), W, stride) + b 

def deconv3d_blk(x, shape, stride, phase):
    """inverse conv3d block with ReLu"""
    W = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=shape[3], initializer=tf.constant_initializer(0.1))
    return deconv3d(tf.nn.relu(tf.contrib.layers.batch_norm(x, is_training=phase)), W, stride) + b 

def res_blk(h_conv1_L, h_conv1_R, shape, stride, phase):
    
    h_conv2_L_a = tf.contrib.layers.batch_norm(h_conv1_L, is_training=phase, scope='bn_a_L') 
    h_conv2_R_a = tf.contrib.layers.batch_norm(h_conv1_R, is_training=phase, scope='bn_a_R') 
    
    with tf.variable_scope("conv_a") as conv2_scope:
        h_conv2_L_b = conv2d_relu(h_conv2_L_a, shape, stride)
        conv2_scope.reuse_variables()
        h_conv2_R_b = conv2d_relu(h_conv2_R_a, shape, stride)  
        
    h_conv3_L_a = tf.contrib.layers.batch_norm(h_conv2_L_b, is_training=phase, scope='bn_b_L') 
    h_conv3_R_a = tf.contrib.layers.batch_norm(h_conv2_R_b, is_training=phase, scope='bn_b_R') 
    
    with tf.variable_scope("conv_b") as conv3_scope:
        h_conv3_L_b = conv2d_relu(h_conv3_L_a, shape, stride)     
        conv3_scope.reuse_variables()
        h_conv3_R_b = conv2d_relu(h_conv3_R_a, shape, stride)  

    h_conv3_L_c = h_conv3_L_b + h_conv1_L
    h_conv3_R_c = h_conv3_R_b + h_conv1_R
        
    return h_conv3_L_c, h_conv3_R_c
        
def cost_volume(img_L, img_R, d_size):
    """
    Cost Volume - each pixel in img_L concat horizontally across img_R
    """
    d = int(d_size/2 - 1)
    dp_list = []

    # when disparity is 0
    elw_tf = tf.concat([img_L, img_R], 3)
    dp_list.append(elw_tf)
    
    # right side
    for dis in range(d):
        # moving the features by disparity d can be done by padding zeros
        pad = tf.constant([[0,0],[0,0],[dis+1,0],[0,0]], dtype=tf.int32)
        pad_R = tf.pad(img_R[:, :, :-1-dis, :], pad, "CONSTANT")
        elw_tf = tf.concat([img_L, pad_R], 3)
        dp_list.append(elw_tf)
    
    total_pack_tf = tf.concat(dp_list, 0)
    total_pack_tf = tf.expand_dims(total_pack_tf, 0)
    return total_pack_tf

def GCNet(img_L, img_R, phase, d=192):
    
    with tf.variable_scope("conv1") as conv1_scope:
        h_1_L = conv2d_blk(img_L, [5, 5, 3, 32], 2)
        conv1_scope.reuse_variables()
        h_1_R = conv2d_blk(img_R, [5, 5, 3, 32], 2)  
    
    with tf.variable_scope("res2-3"):
        h_3_L, h_3_R = res_blk(h_1_L, h_1_R, [3, 3, 32, 32], 1, phase)
        
    with tf.variable_scope("res4-5"):
        h_5_L, h_5_R = res_blk(h_3_L, h_3_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("res6-7"):
        h_7_L, h_7_R = res_blk(h_5_L, h_5_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("res8-9"):
        h_9_L, h_9_R = res_blk(h_7_L, h_7_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("res10-11"):
        h_11_L, h_11_R = res_blk(h_9_L, h_9_R, [3, 3, 32, 32], 1, phase)
 
    with tf.variable_scope("res12-13"):
        h_13_L, h_13_R = res_blk(h_11_L, h_11_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("res14-15"):
        h_15_L, h_15_R = res_blk(h_13_L, h_13_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("res16-17"):
        h_17_L, h_17_R = res_blk(h_15_L, h_15_R, [3, 3, 32, 32], 1, phase)

    with tf.variable_scope("conv18") as conv18_scope:
        h_18_L = conv2d_relu(h_17_L, [3, 3, 32, 32], 1)     
        conv18_scope.reuse_variables()
        h_18_R = conv2d_relu(h_17_R, [3, 3, 32, 32], 1)  
        
    corr = cost_volume(h_18_L, h_18_R, d)

    with tf.variable_scope("conv19"):
        h_19 = conv3d_blk(corr, [3, 3, 3, 64, 32], 1, phase)
    
    with tf.variable_scope("conv20"):
        h_20 = conv3d_blk(h_19, [3, 3, 3, 32, 32], 1, phase)
            
    with tf.variable_scope("conv21"):
        h_21 = conv3d_blk(corr, [3, 3, 3, 64, 64], 2, phase)
        
    with tf.variable_scope("conv22"):
        h_22 = conv3d_blk(h_21, [3, 3, 3, 64, 64], 1, phase)
        
    with tf.variable_scope("conv23"):
        h_23 = conv3d_blk(h_22, [3, 3, 3, 64, 64], 1, phase) 
        
    with tf.variable_scope("conv24"):
        h_24 = conv3d_blk(h_21, [3, 3, 3, 64, 64], 2, phase)  

    with tf.variable_scope("conv25"):
        h_25 = conv3d_blk(h_24, [3, 3, 3, 64, 64], 1, phase)          
        
    with tf.variable_scope("conv26"):
        h_26 = conv3d_blk(h_25, [3, 3, 3, 64, 64], 1, phase)      
        
    with tf.variable_scope("conv27"):
        h_27 = conv3d_blk(h_24, [3, 3, 3, 64, 64], 2, phase)  
        
    with tf.variable_scope("conv28"):
        h_28 = conv3d_blk(h_27, [3, 3, 3, 64, 64], 1, phase)    
        
    with tf.variable_scope("conv29"):
        h_29 = conv3d_blk(h_28, [3, 3, 3, 64, 64], 1, phase)  
        
    with tf.variable_scope("conv30"):
        h_30 = conv3d_blk(h_27, [3, 3, 3, 64, 128], 2, phase)  
        
    with tf.variable_scope("conv31"):
        h_31 = conv3d_blk(h_30, [3, 3, 3, 128, 128], 1, phase)  
        
    with tf.variable_scope("conv32"):
        h_32 = conv3d_blk(h_31, [3, 3, 3, 128, 128], 1, phase)  
        
    with tf.variable_scope("deconv33"):
        h_33_a = deconv3d_blk(h_32, [3, 3, 3, 64, 128], 2, phase)  
        h_33_b = h_33_a + h_29
        
    with tf.variable_scope("deconv34"):
        h_34_a = deconv3d_blk(h_33_b, [3, 3, 3, 64, 64], 2, phase)  
        h_34_b = h_34_a + h_26
        
    with tf.variable_scope("deconv35"):
        h_35_a = deconv3d_blk(h_34_b, [3, 3, 3, 64, 64], 2, phase)  
        h_35_b = h_35_a + h_23
        
    with tf.variable_scope("deconv36"):
        h_36_a = deconv3d_blk(h_35_b, [3, 3, 3, 32, 64], 2, phase)  
        h_36_b = h_36_a + h_20
        
    with tf.variable_scope("conv37"):
        h_37 = deconv3d_blk(h_36_b, [3, 3, 3, 1, 32], 2, phase)
                     
    sqz = tf.squeeze(h_37, 4)
    
    trans = tf.transpose(sqz, perm=[0, 2, 3, 1])
    
    neg = tf.negative(trans)
    logits = tf.nn.softmax(neg)
	
    disparity_filter = tf.reshape(tf.range(0, d, 1, dtype=tf.float32), [1, 1, d, 1])
    distrib = conv2d(logits, disparity_filter, 1)
    return distrib 