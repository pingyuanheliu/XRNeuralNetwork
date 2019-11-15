#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:50:31 2019

@author: ll
"""

import numpy
from PIL import Image

test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

i = 0
# go through all the records in the test data set
for record in test_data_list :
    # split the record by the ',' commas
    all_values = record.split(',')
    # array change to image
    img_array = 255 - numpy.asfarray(all_values[1:]).reshape((28,28))
    img = Image.fromarray(img_array)
    img = img.convert('RGB')
    name = "{}{}{}{}{}".format("test/",i,"_",all_values[0],".png")
    print("name:",name)
    img.save(name)
    i = i+1
    pass