# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:33:11 2018

@author: Kel
"""

class Params():
    def __init__(self):
        self.batch_size = 1
        self.target_h = 256
        self.target_w = 512
        
        self.original_h = 540
        self.original_w = 960
        
        self.max_disparity = 192
        
        self.enqueue_many_size = 200