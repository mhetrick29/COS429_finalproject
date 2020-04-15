###############################
# Learned how to read in file from website: 
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
#
# Based CNN architecture off of this Keras Sequential tutorial:
# https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
# 
###############################

import tensorflow as tf
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping

#method can be used to train our CNN on a data set
def ConvNN():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Format data for keras API; this data-preparation section advised by linked website
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    img_size = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    #Use sequential model because based on research that is what seems like is the best for classification CNN
    model = Sequential()

    #Perform initial convolution. Activation just linear. 3x3 conv-> valid padding would reduce size to 26x26
    #adding more filters takes longer, inc acc
    #Use 3x3 because research suggested 3x3 or 5x5 and our img size was only 28x28
    model.add(Conv2D(16, kernel_size=(3,3), padding="same", input_shape=img_size))

    #Perform 2nd conv
    #Adding more layers takes longer, inc acc
    model.add(Conv2D(32, (3, 3), padding="same", activation = "relu"))
    #Pooling for feature control for not perfectly aligned data set
    #Use max pooling because that's what was recommended in Assignment 4 and articles suggested it
    #Use pooling to downsample more robustly than stride increase
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) #dropout for regularization; .25 kind of arbitrary as rate of disassoc.
    model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))
    model.add(Dropout(0.25))

    #Flatten array for fully connected layers 
    model.add(Flatten())

    #Do fc layer with 256 nodes; would do 14x14x64 but that had estimated training time of a day
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    #Adding another fully conn (like 128 or 64) doesnt help and makes training super long
    model.add(Dense(10,activation="softmax"))

    #Use adam optimizer and sparse crossentropy loss function based on research. Cross-ent better for classification
    #Sparse does not require hot encoding, ie that we necessarily hardcode our 10 classes in memory
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Stop if doesnt imporove after 3 epochs
    checkImprovement = EarlyStopping(patience=3)

    #Data small enough so fit_generator not rqd
    model.fit(x=x_train,y=y_train, validation_split=.2, epochs=10, callbacks=[checkImprovement])

    # model.evaluate(x_test, y_test)

    return model