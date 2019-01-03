from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation
def model(loss='mse', optimizer='adam'):
    """
    The Nvidia model modified to add BatchNormalization.
    The architecture is slightly modified to add batch normalization layers instead of Dropout before ReLU activations are applied.
    
    """
    model = Sequential()

    # Normalization Layer
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(70, 160, 3)))
    
    # Convolutional Layer 1
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 3
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 5
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flatten Layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1))

    # Configure learning process with an optimizer and loss function
    model.compile(loss=loss, optimizer=optimizer)

    return model