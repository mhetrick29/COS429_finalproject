import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from predictNumber import predictNumber
from PN import predictNumber
from DrawBoxes import getBoxes
from solveEquation import solveEquation

#load image
img = cv2.imread('9-8_medium_a.png')

# box and output numbers
num1, op, num2 = getBoxes(img)

# output final equation, solved
one,two,answer = solveEquation(num1,op,num2)

# Print answer
print(one,op,two,"=",answer)

