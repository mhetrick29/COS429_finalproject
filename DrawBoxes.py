import matplotlib  as plt
import cv2
import numpy as np

def getBoxes(img):
    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 3
    val = 100

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    low_threshold = val
    detected_edges = cv2.Canny(gray, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:,:,None].astype(img.dtype))

    is_a_number = 0
    is_in_middle = 0

    top_r = 10000
    bot_r = 0
    prev_c = 0

    height = img.shape[0]
    width = img.shape[1]

    padding_r = 0
    padding_c = 0
    sign = 'plus'

    symbol = 0
    components = []

    np.savetxt('text.txt',detected_edges,fmt='%.2f')
    
    for column in range(0, width):
        is_in_middle = 0
        for row in range(0, height):
            if (detected_edges[row, column] > 30):
                is_in_middle = 1
                if row > bot_r:
                    bot_r = row
                if row < top_r:
                    top_r = row


        if (is_in_middle == 1 and is_a_number == 0):
            is_a_number = 1       
            prev_c = column

        if is_in_middle == 0 and is_a_number == 1:
            if (column - prev_c > bot_r - top_r):
                padding_r = (int) (((column - prev_c) - (bot_r - top_r)) / 2)
            else:
                padding_c = (int) (((bot_r - top_r) - (column - prev_c)) / 2)

            padding_c += 20
            padding_r += 20
            symbol += 1
            #img = cv2.rectangle(img,(prev_c - padding_c, top_r - padding_r),(column + padding_c, bot_r + padding_r),(255,0,0),5)
            is_a_number = 0

            components.append(img[top_r - padding_r:bot_r + padding_r, prev_c - padding_c:column + padding_c])
            
            # save boundaries for +/- Classification 
            if(symbol == 2):
                leftC = prev_c
                rightC = column
                bottomR = bot_r
                topR = top_r

                # +/- DETECTION USING CANNY EDGES 
                colDim = rightC - leftC
                rowDim = topR - bottomR

                ratio = abs(colDim) /abs(rowDim) 
                if (ratio > 2.5):
                    sign = 'minus'

            top_r, bot_r, padding_c, padding_r = 10000, 0, 0, 0

    components[1] = sign
    return components[0],components[1],components[2]