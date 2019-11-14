#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:02:16 2019

@author: ll
"""

import numpy
import csv
import cv2
import imageio
import scipy.special
from PIL import Image
from matplotlib import pyplot as plt

# neural network class definition
class neuralNetwork :
    
    # initialise the neural network
    def __init__(self) :
        
        # link weight matrices, wih and who
        # wih: link weight matrice between input and hidden
        # who: link weight matrice between hidden and output
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = []
        with open("mnist_dataset/wih.csv","r") as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                newRow = []
                for var in row:
                    newRow.append(float(var))
                    pass
                self.wih.append(newRow)
                pass
            pass
        #
        self.who = []
        with open("mnist_dataset/who.csv","r") as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                newRow = []
                for var in row:
                    newRow.append(float(var))
                    pass
                self.who.append(newRow)
                pass
            pass
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # query the neural network
    def query(self, inputs_list) :
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    pass

# create instance of neural network
n = neuralNetwork()

test_data_list = []

# load image data from png files into an array
img = cv2.imread('test_images/01_2.png',0)
img = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# show threshold image
plt.subplot(2,3,1),plt.imshow(img,'gray')
plt.title('TRUNC')
plt.xticks([]),plt.yticks([])
plt.show()
# reshape from 28x28 to list of 784 values, invert values
img_data  = 255 - img.reshape(784)
print(numpy.min(img_data))
print(numpy.max(img_data))
# append label and image data  to test data set
record = numpy.append(2,img_data)
test_data_list.append(record)

# test the neural network

# scorecard for how well the network peforms, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list :
    # correct answer is first value
    correct_label = int(record[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(record[1:]) / 255.0 * 0.99)
    # query the network
    outputs = n.query(inputs)
    print("outputs:")
    print(outputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("label:",label)
    # append correct or incorrect to list
    if (label == correct_label) :
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
        pass
    else :
        # network's answer doesn't match correct answer add 0 to scorecard
        scorecard.append(0)
        pass
    pass
pass
# calculate the performance scored, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
print("scorecard",scorecard)