from keras.layers import Conv3D, MaxPooling3D
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.layers import TimeDistributed, Flatten, LSTM, Dense, Activation, ZeroPadding2D, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from keras.layers import MaxPool2D, GlobalMaxPool2D
#import tensorflow as tf
import logging
from data_types import *


def lstm_model(X_train_shape, parameters):
    #print(parameters)
    hidden_units = int(parameters[0])
    dropout_parameter = float(parameters[1])
    loss_function = str(parameters[2])
    optimizer = str(parameters[3])
    lstm_cells = int(parameters[4])
    dropout = float(parameters[5])
    cnn_flattened_layer_1 = int(parameters[6])
    cnn_flattened_layer_2 = int(parameters[7])
    #lr = int(parameters[8])

    print('Build model...')
    print('X_train shape: ', X_train_shape)
    print('Final layer', DATA_SET_INFO['num_classes'])

    input_shape = (int((DATA_SET_INFO['num_classes']+ 2) /2), DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])

    logging.info("Architecture:%s,%s,%s,%s,%s" % (hidden_units, dropout_parameter, loss_function, optimizer, lstm_cells))

    # define the CNN model
    # create cnn_model to train
    cnn_model = Sequential()

    # Add our first convolutional layer
    cnn_model.add(Conv2D(filters=64,
                   kernel_size=(9, 9),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1'))
    cnn_model.add(Conv2D(filters=32,
                    kernel_size=(5, 5),
                    padding='valid',
                    strides=(2, 2),
                    activation='relu',
                    name='conv2'))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv3'))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv4'))
    #cnn_model.add(BatchNormalization())
    #cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    cnn_model.add(Flatten(name='flat'))

    # # Add Fully connected ANN
    cnn_model.add(Dense(units=cnn_flattened_layer_1, activation='relu', name='fc6'))
    cnn_model.add(Dropout(dropout))
    cnn_model.add(Dense(units=cnn_flattened_layer_2, activation='relu', name='fc7', ))
    cnn_model.add(Dropout(dropout))
    # cnn_model.add(Dense(units=int(num_categories), activation='softmax', name='fc8'))

    model = Sequential()
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))

    if lstm_cells == 1:
            model.add((LSTM(hidden_units, return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 2:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 3:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 4:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 5:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units), return_sequences=False)))
            model.add(Dropout(dropout_parameter))

    model.add(Dense(units=256, activation='relu'))
    model.add(dropout)
    model.add(Dense(units=128, activation='relu'))
    model.add(dropout)
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='relu'))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mse'])


    print(model.summary())


    return model



def lstm_test(X_train_shape, parameters):
    #print(parameters)
    #lr = int(parameters[8])

    print('Build model...')
    print('X_train shape: ', X_train_shape)
    print('Final layer', DATA_SET_INFO['num_classes'])

    input_shape = (int((DATA_SET_INFO['num_classes']+ 2) /2), DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])

    #logging.info("Architecture:%s,%s,%s,%s,%s" % (hidden_units, dropout_parameter, loss_function, optimizer, lstm_cells))

    # define the CNN model
    # create cnn_model to train
    cnn_model = Sequential()

    #convnet = build_convnet(input_shape[1:])

    #input_shape=(5, 224, 224, 3)
    cnn_model.add((Conv2D(filters=64,
                   kernel_size=(9, 9),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1')))
    cnn_model.add(BatchNormalization(momentum=0.9))
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))

    # Add second convolutional layer.
    cnn_model.add(ZeroPadding2D(padding=(2, 2)))
    cnn_model.add(Conv2D(filters=32,
                          kernel_size=(5, 5),
                          padding='valid',
                          strides=(2, 2),
                          activation='relu',
                          name='conv2'))
    cnn_model.add(BatchNormalization(momentum=0.9))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv3'))
    cnn_model.add(BatchNormalization(momentum=0.9))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv4'))
    cnn_model.add(BatchNormalization(momentum=0.9))
    #cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    cnn_model.add(Flatten(name='flat'))
    # Add Fully connected ANN
    cnn_model.add(Dense(units=1024, activation='relu', name='fc6'))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(Dense(units=512, activation='relu', name='fc7', ))
    cnn_model.add(Dropout(0.1))
    # cnn_model.add(Dense(units=int(num_categories), activation='softmax', name='fc8'))

    model = Sequential()
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))

    model.add((LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    #model.add(Dropout(0.2))

    model.add((LSTM(256, return_sequences=False)))
    model.add(Dropout(0.2))
        #model.add(Dropout(0.1))

    model.add(Dense(units=256, activation='relu'))
    cnn_model.add(Dropout(0.1))
    model.add(Dense(units=128, activation='relu'))
    cnn_model.add(Dropout(0.1))
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='relu'))

    # Set Optimizer
    #opt = adam_v2.Adam(lr=0.0001, decay=0.0001)
    model.compile(loss='logcosh', optimizer='adam', metrics=['mse'])

    print("*** Test Model ***")
    print(model.summary())




    return model
