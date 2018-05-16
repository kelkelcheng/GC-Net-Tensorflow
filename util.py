# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:24:05 2018

@author: Kel
"""
import tensorflow as tf

def read_and_decode(params, filename):
    width, height = params.original_w, params.original_h
    batch_size = params.batch_size
    target_w, target_h  = params.target_w, params.target_h

    filename_queue = tf.train.string_input_producer([filename]) 
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
    serialized_example,

    features={
      'img_left': tf.FixedLenFeature([], tf.string),
      'img_right': tf.FixedLenFeature([], tf.string),
      'disparity': tf.FixedLenFeature([], tf.string)
    })
    

    image_left = tf.decode_raw(features['img_left'], tf.uint8)
    image_left= tf.reshape(image_left, [height, width, 3])
    
    image_right = tf.decode_raw(features['img_right'], tf.uint8)
    image_right = tf.reshape(image_right, [height, width, 3])
  
    disparity = tf.decode_raw(features['disparity'], tf.float32)
    disparity = tf.reshape(disparity, [height, width, 1])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_left = tf.cast(image_left, tf.float32) * (1. / 255) - 0.5
    image_right = tf.cast(image_right, tf.float32) * (1. / 255) - 0.5

    concat = tf.concat([image_left, image_right, disparity], 2)
    img_crop = tf.random_crop(concat, [target_h, target_w, 7])
    
    image_left_batch, image_right_batch, disparity_batch = tf.train.shuffle_batch([img_crop[:,:,0:3], img_crop[:,:,3:6], img_crop[:,:,6:]],
                                                                                  batch_size=batch_size, capacity=50,
                                                                                  min_after_dequeue=10, num_threads=2)
 
    return [image_left_batch, image_right_batch, disparity_batch]