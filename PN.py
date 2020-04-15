###############################
# Takes in an image of handwritten number and uses trained CNN to predict the number
# Logical flow based on this tutorial about positioning images for MNIST:
# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
## And the MNIST site on how they prepared the data (http://yann.lecun.com/exdb/mnist/) 
###############################


import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy import ndimage
import tensorflow as tf
from keras.models import load_model
from keras.layers import Activation

def predictNumber(img):
    model = load_model("Model_3conv.h5", custom_objects = {'softmax_v2': tf.nn.softmax})
    #convert to grayscale and invert to match MNIST set
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = 255-img_gray
    img_gray = cv2.resize(img_gray, (28, 28))
    #plt.imshow(img_gray, cmap = "Greys")

    #perform thresholding. Tried to do a different way but article, MNIST guide, and googling said to use threshold
    #use binary thresholding based on cv2 documentation
    retVal, img_gray = cv2.threshold(img_gray, 115, 255, cv2.THRESH_BINARY)

    #shave edges to fit to 20x20
    #logic for this attributed to medium website linked in header

    ##########################################################################
    # This section between the lines is from Medium (towardsdatascience) website tutorial on image
    # Used tutorials and code based on TA recommendation to preprocess images (thank you!)
    # We simply changed the parameters passed in to fit our project
    
    # trim top edge
    while np.sum(img_gray[0]) == 0:
        img_gray = img_gray[1:]

    # trim left edge
    while np.sum(img_gray[:,0]) == 0:
        img_gray = np.delete(img_gray,0,1)
    
    # trim bottom edge
    while np.sum(img_gray[-1]) == 0:
        img_gray = img_gray[:-1]

    # trim right edge
    while np.sum(img_gray[:,-1]) == 0:
        img_gray = np.delete(img_gray,-1,1)

    # center in 28x28 frame
    # whichever side is larger, scale that by a factor of 20/side to obtain tightly fit image
    # this essentially fits border of image to the number
    rows, cols = img_gray.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img_gray = cv2.resize(img_gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img_gray = cv2.resize(img_gray, (cols,rows))

    # adds padding to make the image 28x28
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img_gray = np.lib.pad(img_gray,(rowsPadding,colsPadding),'constant')
    ###################################################################################

    # shift using scipy documentation about how to center an image
    # get center of mass and size
    yBar,xBar = ndimage.measurements.center_of_mass(img_gray)
    size = img_gray.shape

    # get how much we need to move image and perform affine transformation
    # formatting from opencv tutorial of how to use warpAffine
    moveLR = np.round(size[1]/2.0-xBar).astype(int)
    moveUD = np.round(size[0]/2.0-yBar).astype(int)
    M = np.float32([[1,0,moveLR],[0,1,moveUD]])
    img_gray = cv2.warpAffine(img_gray,M,(size[1],size[0]))
    
    # format for keras model
    img_gray = img_gray/255
    predictionImage = img_gray.reshape(1, 28, 28, 1)
    
    #visualize image
    #plt.imshow(predictionImage.reshape(28,28), cmap = "Greys")

    pred = model.predict(predictionImage)
    
    return pred.argmax()