import cv2
import numpy as np
import os
import math
from scipy import ndimage
import tensorflow as tf
from keras.models import load_model
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def imageResize(image, height):
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim)
    return resized

def main():

    # read image and resize it
    image = cv2.imread("image.JPG")
    image = cv2.resize(image, (280,280))

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image to smooth outliers
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # apply thresholding to differentiate between foreground and background
    thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV)[1]

    # remove excess padding around number
    x, y, w, h = cv2.boundingRect(thresh)
    box = thresh[y:y + h, x:x + w]

    rows,cols = box.shape

    # resize image dimensions to fit within a 20x20 pixel box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        box20 = cv2.resize(box, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        box20 = cv2.resize(box, (cols, rows))

    # add padding to create a 28x28 pixel box
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    box28 = np.lib.pad(box20,(rowsPadding,colsPadding),'constant')

    # find center of mass of image and calculate the shift values for the x and y axis
    centerY,centerX = ndimage.measurements.center_of_mass(box28)
    rows,cols = box28.shape
    shiftX = np.round(cols/2.0-centerX).astype(int)
    shiftY = np.round(rows/2.0-centerY).astype(int)

    # shift image so that it conforms to the center of mass
    M = np.float32([[1,0,shiftX],[0,1,shiftY]])
    center = cv2.warpAffine(box28,M,(cols,rows))

    # load trained model with all weights and parameters
    model = load_model('mnist.h5')

    # reshape image to 4D array
    final = center.reshape(1, 28, 28, 1)
    final = final.astype('float32')
    final /= 255

    # make prediction
    pred = model.predict(final)
    print(pred)
    print(pred.argmax())

if __name__ == "__main__":
    main()