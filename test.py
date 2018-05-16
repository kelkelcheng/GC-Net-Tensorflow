# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:06:31 2017

@author: Kel
"""

import tensorflow as tf
import numpy as np
from PIL import Image

import graph
import params
import util

train_dir = 'saved_model/'

data_record = ["../fly_train.tfrecords", "../fly_test.tfrecords"]
        
p = params.Params()    
    
batch_train = util.read_and_decode(p, data_record[0])
batch_test = util.read_and_decode(p, data_record[1])

img_L = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 3])
img_R = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 3])
disp = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 1])
phase = tf.placeholder(tf.bool)

pred = graph.GCNet(img_L, img_R, phase, p.max_disparity)

#loss = tf.reduce_mean(tf.losses.mean_squared_error(pred, gt))
loss = tf.losses.absolute_difference(pred, disp)

learning_rate = 0.001
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

global_step = tf.Variable(0, name='global_step', trainable=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)
    
init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

saver = tf.train.Saver() 

img_path = "middlebury/flower/"
with tf.Session() as sess:
    restore_dir = tf.train.latest_checkpoint(train_dir)
    if restore_dir:
        saver.restore(sess, restore_dir)
        print('restore succeed')
    else:
        sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

     # Convert from [0, 255] -> [-0.5, 0.5] floats.
    img_1 = np.asarray(Image.open(img_path+"im0.png").resize((p.target_w, p.target_h))) * (1. / 255) - 0.5
    img_2 = np.asarray(Image.open(img_path+"im1.png").resize((p.target_w, p.target_h))) * (1. / 255) - 0.5

    batch = sess.run(batch_test)
    feed_dict = {img_L: [img_1], img_R: [img_2], phase: False}
    [f_out] = sess.run([pred], feed_dict=feed_dict)

    im_out = Image.fromarray(np.reshape(f_out, (p.target_h, p.target_w))/191.0*255.0).convert('RGB')

    im_out.show()
    im_out.save('output_img/test_img.jpg')

    coord.request_stop()
    coord.join(threads)