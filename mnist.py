import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import os

def main():

    # split training and testing set data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to 4D array
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    #Convert to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the RGB codes
    x_train /= 255
    x_test /= 255

    input_shape = (28, 28, 1)

    # Build sequential model and add the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    
    # use adam optimizer, crossentropy loss function and accuracy metric to compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # fit the model with training data
    model.fit(x=x_train,y=y_train, epochs=10)

    # evaluate trained model with x_test and y_test
    print(model.evaluate(x_test, y_test))

    # create a HDF5 file
    model.save("mnist.h5")

if __name__ == "__main__":
    main()