from keras.layers import Conv3D, MaxPooling3D
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.regularizers import l2
from keras.layers import TimeDistributed, Flatten, LSTM, Dense, Activation, ZeroPadding2D, Dropout, BatchNormalization, RepeatVector, Bidirectional
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from keras.layers import MaxPool2D, GlobalMaxPool2D
import pandas as pd
#import tensorflow as tf

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from keras import backend as K

import logging
from data_types import *


def lstm_model(X_train_shape, parameters):
    #print(parameters)
    hidden_units = int(parameters[0])
    dropout_parameter = float(parameters[1])
    momentum = float(parameters[2])
    loss_function = str(parameters[3])
    optimizer = str(parameters[4])
    lstm_cells = int(parameters[5])
    dropout = float(parameters[6])
    cnn_flattened_layer_1 = int(parameters[7])
    cnn_flattened_layer_2 = int(parameters[8])
    lstm_flattened_layer_1 = int(parameters[9])
    lstm_flattened_layer_2 = int(parameters[10])
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
    cnn_model.add((Conv2D(filters=64,
                   kernel_size=(9, 9),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1')))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))

    # Add second convolutional layer.
    cnn_model.add(ZeroPadding2D(padding=(2, 2)))
    cnn_model.add(Conv2D(filters=32,
                          kernel_size=(5, 5),
                          padding='valid',
                          strides=(2, 2),
                          activation='relu',
                          name='conv2'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv3'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv4'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    #cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
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

    model.add(Dense(units=lstm_flattened_layer_1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=lstm_flattened_layer_2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='relu'))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mse'])


    print(model.summary())


    return model

def lstm_autoencoder_model(X_train_shape, parameters):
    #print(parameters)
    hidden_units = int(parameters[0])
    dropout_parameter = float(parameters[1])
    momentum = float(parameters[2])
    loss_function = str(parameters[3])
    optimizer = str(parameters[4])
    lstm_cells = int(parameters[5])
    dropout = float(parameters[6])
    cnn_flattened_layer_1 = int(parameters[7])
    cnn_flattened_layer_2 = int(parameters[8])
    lstm_flattened_layer_1 = int(parameters[9])
    lstm_flattened_layer_2 = int(parameters[10])
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
    cnn_model.add((Conv2D(filters=64,
                   kernel_size=(9, 9),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1')))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))

    # Add second convolutional layer.
    cnn_model.add(ZeroPadding2D(padding=(2, 2)))
    cnn_model.add(Conv2D(filters=32,
                          kernel_size=(5, 5),
                          padding='valid',
                          strides=(2, 2),
                          activation='relu',
                          name='conv2'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv3'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    cnn_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv4'))
    cnn_model.add(BatchNormalization(momentum=momentum))
    #cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    cnn_model.add(Flatten(name='flat'))


    model = Sequential()
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))


    if lstm_cells == 1:
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False)))
        model.add(Dropout(dropout_parameter))
        model.add(RepeatVector(DATA_SET_INFO['num_classes']))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
    elif lstm_cells == 2:
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False)))
        model.add(Dropout(dropout_parameter))
        model.add(RepeatVector(DATA_SET_INFO['num_classes']))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
    elif lstm_cells == 3:
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=False)))
        model.add(Dropout(dropout_parameter))
        model.add(RepeatVector(DATA_SET_INFO['num_classes']))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_parameter))


    model.add(Dense(units=lstm_flattened_layer_1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=lstm_flattened_layer_2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='relu'))

    model.add(TimeDistributed(Dense(1)))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mse'])


    print(model.summary())


    return model

def genetic_CNN_lstm_test(X_train_shape, parameters, encoding):
    hidden_units = int(parameters[0])
    dropout_parameter = float(parameters[1])
    momentum = float(parameters[2])
    loss_function = str(parameters[3])
    optimizer = str(parameters[4])
    lstm_cells = int(parameters[5])
    dropout = float(parameters[6])
    cnn_flattened_layer_1 = int(parameters[7])
    cnn_flattened_layer_2 = int(parameters[8])
    lstm_flattened_layer_1 = int(parameters[9])
    lstm_flattened_layer_2 = int(parameters[10])

    print('Build model')
    print('X_train shape: ', X_train_shape)
    print('Final layer', DATA_SET_INFO['num_classes'])

    input_shape = (int((DATA_SET_INFO['num_classes']+ 2) /2), DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])

def lstm_test(X_train_shape, parameters):
    #print(parameters)
    #lr = int(parameters[8])

    hidden_units = int(parameters[0])
    dropout_parameter = float(parameters[1])
    momentum = float(parameters[2])
    loss_function = str(parameters[3])
    optimizer = str(parameters[4])
    lstm_cells = int(parameters[5])
    dropout = float(parameters[6])
    cnn_flattened_layer_1 = int(parameters[7])
    cnn_flattened_layer_2 = int(parameters[8])
    lstm_flattened_layer_1 = int(parameters[9])
    lstm_flattened_layer_2 = int(parameters[10])

    print('Build model')
    print('X_train shape: ', X_train_shape)
    print('Final layer', DATA_SET_INFO['num_classes'])

    input_shape = (int((DATA_SET_INFO['num_classes']+ 2) /2), DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])

    #print('Batch size: ', PARAMETERS_LSTM['batch_size'])

    #logging.info("Architecture:%s,%s,%s,%s,%s" % (hidden_units, dropout_parameter, loss_function, optimizer, lstm_cells))

    # define the CNN model
    # create cnn_model to train
    cnn_model = Sequential()


    # Add second convolutional layer.
    #cnn_model.add(ZeroPadding2D(padding=(2, 2)))
    cnn_model.add(Conv2D(filters=32,
                          kernel_size=(3, 3),
                          padding='valid',
                          strides=(1, 1),
                          data_format='channels_last',
                          input_shape=input_shape[1:],
                          activation='relu',
                          name='conv2'))

    # --- STAGE 1 ---
    #if encoding = '0-00':

    #if encoding = '0-01':

    #if encoding = '0-10':

    #if encoding = '0-11':

    #if encoding = '1-00':

    #if encoding = '1-01':

    #if encoding = '1-10':

    #if encoding = '1-11': 

    #cnn_model.add(BatchNormalization(momentum=0.95))
    #cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    cnn_model.add(Flatten(name='flat'))

    #convnet = build_convnet(input_shape[1:])

    #input_shape=(5, 224, 224, 3)
    #cnn_model.add((Conv2D(filters=64,
    #               kernel_size=(9, 9),
    #               strides=(4, 4),
    #               padding='valid',
    #               data_format='channels_last',
    #               input_shape=input_shape[1:],
    #               activation='relu',
    #               name='conv1')))
    #cnn_model.add(BatchNormalization(momentum=0.95))
    #cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))

    # Add second convolutional layer.
    #cnn_model.add(ZeroPadding2D(padding=(2, 2)))
    cnn_model.add(Conv2D(filters=64,
                          kernel_size=(5, 5),
                          padding='valid',
                          strides=(3, 3),
                          data_format='channels_last',
                          input_shape=input_shape[1:],
                          activation='relu',
                          name='conv2'))
    #cnn_model.add(BatchNormalization(momentum=0.95))
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))
    cnn_model.add(Conv2D(filters=64,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv3'))
    #cnn_model.add(BatchNormalization(momentum=0.95))
    cnn_model.add(Conv2D(filters=64,
                        kernel_size=(3, 3),
                        padding='valid',
                        strides=(1, 1),
                        activation='relu',
                        name='conv4'))
    #cnn_model.add(BatchNormalization(momentum=0.95))
    #cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    cnn_model.add(Flatten(name='flat'))
    # Add Fully connected ANN
    #cnn_model.add(Dense(units=1024, activation='relu', name='fc6'))
    #cnn_model.add(Dropout(0.1))
    #cnn_model.add(Dense(units=512, activation='relu', name='fc7', ))
    #cnn_model.add(Dropout(0.1))
    # cnn_model.add(Dense(units=int(num_categories), activation='softmax', name='fc8'))

    model = Sequential()
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))
    #
    #model.add(Bidirectional(LSTM(200, return_sequences=False)))
    #model.add(Dropout(0.1))
    #
    #model.add(RepeatVector(DATA_SET_INFO['num_classes']))
    #
    #model.add(Bidirectional(LSTM(200, return_sequences=True)))
    #model.add(Dropout(0.1))
    #
    # #model.add(Dropout(0.2))
    #
    model.add((Dense(units=512, activation='relu')))
    cnn_model.add(Dropout(0.1))
    # model.add(TimeDistributed(Dense(1)))

    #model.add(TimeDistributed(Dense(DATA_SET_INFO['num_classes'], activation='relu')))

    model.add(LSTM(250, return_sequences=True, kernel_regularizer=l2(0.0001)))
    model.add(LSTM(250, return_sequences=False, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.1))
    #
    # #model.add(Dropout(0.2))
    #
    model.add(Dense(units=512, activation='relu'))
    cnn_model.add(Dropout(0.1))
    #model.add(TimeDistributed(Dense(1)))

    model.add(Dense(DATA_SET_INFO['num_classes'], activation='relu'))

    #def custom_mean_squared_error(y_true, y_pred):
    #    X_true = y_true[:,1::2]
    #    X_pred = y_pred[:,0::2]
    #    Y_true = y_true[:,1::2]
    #    Y_pred = y_pred[:,0::2]
    #    Y = K.square(Y_pred - Y_true)
    #    X = K.square(X_pred - X_true)
    #    return K.mean(Y + X)

    # Set Optimizer
    #opt = adam_v2.Adam(lr=0.0001, decay=0.0001)
    model.compile(loss='logcosh', optimizer='nadam', metrics=['mse'])

    print("*** Test Model ***")
    print(model.summary())


    return model
