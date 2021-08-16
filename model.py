import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, TimeDistributed, Flatten, LSTM, Dense, Activation, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import logging

from data_types import *

# patience=5)
# monitor='val_loss',patience=2,verbose=0
# In your case, you can see that your training loss is not dropping
# - which means you are learning nothing after each epoch.
# It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.


class MyTemporalLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyTemporalLayer, self).__init__()

  def build(self, input_shape):
    return

  def call(self, inputs):
    return tf.stack(inputs, axis=1, name='stack')



def compile_model_cnn(genome, nb_classes, input_shape):
    # Get our network parameters.
    nb_layers = genome.geneparam['nb_layers']
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer = genome.geneparam['optimizer']

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))


    model = Sequential()

    # Add each layer.
    for i in range(0, nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation, padding='same',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation))

        # otherwise we hit zero
        if i < 2:
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

    model.add(Flatten())
    # always use last nb_neurons value for dense layer
    model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    # need to read this paper
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def dgn_model(X_train_shape, parameters):
    nb_neurons_1st_fc = int(parameters[0])
    nb_neurons_2nd_fc = int(parameters[1])
    loss_function = str(parameters[2])
    optimizer = str(parameters[3])

    print('Build model...')
    print('X_train shape: ', X_train_shape)
    logging.info("Architecture:%s,%s,%s,%s" % (nb_neurons_1st_fc, nb_neurons_2nd_fc, loss_function, optimizer))
    print(nb_neurons_1st_fc, nb_neurons_2nd_fc, loss_function, optimizer)

    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=X_train_shape[1:], kernel_size=(9, 9), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=int(nb_neurons_1st_fc), input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=int(nb_neurons_2nd_fc)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(int(DATA_SET_INFO['num_classes'])))
    model.add(Activation('softmax'))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model


def conv3d_model(X_train_shape, parameters):
    loss_function = str(parameters[0])
    optimizer = str(parameters[1])
    hidden_units = int(parameters[2])
    fc_layers = int(parameters[3])

    print('Build model...')
    print('X_train shape: ', X_train_shape)

    logging.info("Architecture:%s,%s,%s,%s" % (loss_function, optimizer, hidden_units, fc_layers))

    model = Sequential()
    model.add(Conv3D(hidden_units, kernel_size=(3, 3, 3), activation='relu', input_shape=X_train_shape[1:],
                     data_format='channels_last', padding='same'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    #model.add(BatchNormalization())
    model.add(Flatten(name='flat'))
    model.add(Dense(fc_layers, activation='relu', name='fc1'))
    model.add(Dropout(.5))
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='softmax'))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    return model


def lstm_model(X_train_shape, parameters):
    hidden_units = int(parameters[0])
    dropout_parameter = int(parameters[1])
    loss_function = str(parameters[2])
    optimizer = str(parameters[3])
    lstm_cells = int(parameters[4])

    seq_len = int((DATA_SET_INFO['num_classes'] + 2 )/2)

    print('Build model...')
    print('X_train shape: ', X_train_shape)
    print('Final layer', DATA_SET_INFO['num_classes'])

    input_shape = (DATA_SET_INFO['image_channels'], DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])

    logging.info("Architecture:%s,%s,%s,%s,%s" % (hidden_units, dropout_parameter, loss_function, optimizer, lstm_cells))

    # def get_cnn_model(input_shape=(7, 7, 3)):
    #   input_cnn = tf.keras.layers.Input(shape=input_shape)
    #   conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(input_cnn)
    #   model_cnn = tf.keras.Model(inputs=input_cnn, outputs=conv_2d_layer)
    #   print(model_cnn.summary())
    #   return model_cnn

    # define the CNN model
    # create dgn to train
    # dgn = Sequential()
    print("Input shape")
    inputs = tf.keras.layers.Input(input_shape[1:])
    print(input_shape[1:])
    # Add our first convolutional layer
    def get_cnn_model(input_shape_=input_shape[1:]):
        input_cnn = tf.keras.layers.Input(shape=input_shape_)
        x = Conv2D(filters=64,
                   kernel_size=(7, 7),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   #input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1')(input_cnn)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1')(x)

	# Add second convolutional layer.
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = Conv2D(filters=32,
                   kernel_size=(5, 5),
                   strides=(2,2),
                   padding='valid',
                   data_format='channels_last',
                   #input_shape=input_shape[1:],
                   activation='relu',
                   name='conv2')(input_cnn)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2')(x)

        # Add second convolutional layer.
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = Conv2D(filters=32,
                          kernel_size=(3, 3),
                          padding='valid',
                          strides=(1, 1),
                          activation='relu',
                          name='conv3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool3')(x)

        # Add second convolutional layer.
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = Conv2D(filters=32,
                          kernel_size=(3, 3),
                          padding='valid',
                          strides=(1, 1),
                          activation='relu',
                          name='conv4')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool4')(x)

        x = Flatten(name='flat')(x)
        # # Add Fully connected ANN
		# TO CONSIDER: adding these as genetic hyperparameters
        x = Dense(units=1024, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=512, activation='relu', name='fc7', )(x)
        x = Dropout(0.5)(x)

        model_cnn = tf.keras.Model(inputs=input_cnn, outputs=x)
        print(model_cnn.summary())
        return model_cnn
    # dgn.add(Dense(units=int(num_categories), activation='softmax', name='fc8'))

    #model = Sequential()
    #model.add(TimeDistributed(dgn, input_shape=input_shape))
    #dgn = Model(inputs=inputs, outputs=x)
    #x = TimeDistributed(dgn)(Input(input_shape))
    #flatten = Reshape(seq_len, -1)
    cnn_model = get_cnn_model()

    input_seq = [tf.keras.layers.Input(input_shape[1:]) for _ in range(seq_len)]
    cnn_outputs = []

    for i in range(seq_len):
        cnn_outputs.append(cnn_model(input_seq[i]))

    print("cnn_outputs")
    print(cnn_outputs)
    input_lstm = MyTemporalLayer()(cnn_outputs)
    print("input_lstm")
    print(input_lstm)

    flatten = tf.keras.layers.Reshape((seq_len, -1))(input_lstm)
    print(flatten)

    if lstm_cells == 1:
        lstm = LSTM(hidden_units, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
    elif lstm_cells == 2:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
    elif lstm_cells == 3:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
    elif lstm_cells == 4:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
    elif lstm_cells == 5:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
    elif lstm_cells == 6:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=True, dropout=dropout_parameter)
        lstm6 = LSTM(hidden_units*6, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
        output = lstm6(output)
    elif lstm_cells == 7:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=True, dropout=dropout_parameter)
        lstm6 = LSTM(hidden_units*6, return_sequences=True, dropout=dropout_parameter)
        lstm7 = LSTM(hidden_units*7, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
        output = lstm6(output)
        output = lstm7(output)
    elif lstm_cells == 8:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=True, dropout=dropout_parameter)
        lstm6 = LSTM(hidden_units*6, return_sequences=True, dropout=dropout_parameter)
        lstm7 = LSTM(hidden_units*7, return_sequences=True, dropout=dropout_parameter)
        lstm8 = LSTM(hidden_units*8, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
        output = lstm6(output)
        output = lstm7(output)
        output = lstm8(output)
    elif lstm_cells == 9:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=True, dropout=dropout_parameter)
        lstm6 = LSTM(hidden_units*6, return_sequences=True, dropout=dropout_parameter)
        lstm7 = LSTM(hidden_units*7, return_sequences=True, dropout=dropout_parameter)
        lstm8 = LSTM(hidden_units*8, return_sequences=True, dropout=dropout_parameter)
        lstm9 = LSTM(hidden_units*9, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
        output = lstm6(output)
        output = lstm7(output)
        output = lstm8(output)
        output = lstm9(output)
    elif lstm_cells == 10:
        lstm = LSTM(hidden_units, return_sequences=True, dropout=dropout_parameter)
        lstm2 = LSTM(hidden_units*2, return_sequences=True, dropout=dropout_parameter)
        lstm3 = LSTM(hidden_units*3, return_sequences=True, dropout=dropout_parameter)
        lstm4 = LSTM(hidden_units*4, return_sequences=True, dropout=dropout_parameter)
        lstm5 = LSTM(hidden_units*5, return_sequences=True, dropout=dropout_parameter)
        lstm6 = LSTM(hidden_units*6, return_sequences=True, dropout=dropout_parameter)
        lstm7 = LSTM(hidden_units*7, return_sequences=True, dropout=dropout_parameter)
        lstm8 = LSTM(hidden_units*8, return_sequences=True, dropout=dropout_parameter)
        lstm9 = LSTM(hidden_units*9, return_sequences=True, dropout=dropout_parameter)
        lstm10 = LSTM(hidden_units*10, return_sequences=False, dropout=dropout_parameter)
        output = lstm(flatten)
        output = lstm2(output)
        output = lstm3(output)
        output = lstm4(output)
        output = lstm5(output)
        output = lstm6(output)
        output = lstm7(output)
        output = lstm8(output)
        output = lstm9(output)
        output = lstm10(output)


	# TO CONSIDER: adding these as genetic hyperparameters
    output = Dense(units=128, activation='relu', name='fc_lstm1')(output)
    output = Dense(units=128, activation='relu', name='fc_lstm2')(output)
    out = Dense(DATA_SET_INFO['num_classes'], activation='softmax')(output)
    model = Model(inputs=input_seq, outputs=out)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())




    return model
