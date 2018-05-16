import os
import tensorflow as tf
from PIL import Image
import re
import numpy as np
from scipy.misc import imresize

def readPFM(file):
    file = open(file, 'r', encoding='ISO-8859-1')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def rgba_to_rgb(img):
    '''
    change image from rgba to rgb
    [height, width, 4] -> [height, width, 3]
    '''    
    img.load()
    img_temp = Image.new("RGB", img.size, (255,255,255))
    img_temp.paste(img, mask=img.split()[3])
    return img_temp
    
cwd = os.getcwd()
dirs = [cwd + '/' + 'flyingthings3d_frames_cleanpass/',
        cwd + '/' + 'flyingthings3d__disparity/disparity/']

writer_tr = tf.python_io.TFRecordWriter("fly_train.tfrecords")
writer_ts = tf.python_io.TFRecordWriter("fly_test.tfrecords")

count = 0
for phase in ['TRAIN', 'TEST']:
    for group in ['A', 'B', 'C']:
        dir_group = dirs[0] + phase + '/' + group
        dir_group2 = dirs[1] + phase + '/' + group
        for img_group in os.listdir(dir_group):
            dir_img_group = dir_group + '/' + img_group
            dir_dis_group = dir_group2 + '/' + img_group
            for img_name in os.listdir(dir_img_group + '/left'):         
                img_path_1 = dir_img_group + '/left/' + img_name
                img_1 = Image.open(img_path_1)
                #img_1 = img_1.resize((width, height))
                #img_1 = rgba_to_rgb(img_1)
                img_1 = np.array(img_1)
                img_1_raw = img_1.tobytes()
                
                img_path_2 = dir_img_group + '/right/' + img_name
                img_2 = Image.open(img_path_2)
                #img_2 = img_2.resize((width, height))
                #img_2 = rgba_to_rgb(img_2)
                img_2 = np.array(img_2)
                img_2_raw = img_2.tobytes()
                
                disparity_path = dir_dis_group + '/left/' + img_name.split('.')[0] + '.pfm'
                disparity = readPFM(disparity_path)[0]
                disparity_raw = disparity.tobytes()
                        
                example = tf.train.Example(features=tf.train.Features(feature={
                                 "img_left": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_1_raw])),
                                 'img_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_2_raw])),
                                 'disparity': tf.train.Feature(bytes_list=tf.train.BytesList(value=[disparity_raw]))}))
                 
                count += 1
                if phase == 'TRAIN':
                    writer_tr.write(example.SerializeToString())
                else:
                    writer_ts.write(example.SerializeToString())

 
writer_tr.close()
writer_ts.close()
