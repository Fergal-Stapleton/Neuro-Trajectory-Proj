"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
import sys
from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
from genetic_algorithm import GeneticAlgorithm
from grid_search import GridSearch
from keras import backend as K
from data_types import *
from load_data import *
import time
import logging
import model
import copy

# Setup logging.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    filename='./log.txt')

def load_data(MODEL_NAME):
    data_set = LoadData(MODEL_NAME, DATA_SET_INFO['data_set_path'], DATA_SET_INFO['image_height'],
                        DATA_SET_INFO['image_width'], DATA_SET_INFO['image_channels'],
                        DATA_SET_INFO['image_depth'], DATA_SET_INFO['num_classes'])

    if data_set.absolute_path_cond == False:
        path_to_npy = './data_sets/' + MODEL_NAME + '/X_train.npy'
    else:
        path_to_npy = data_set.absolute_path + 'data_sets/' + MODEL_NAME + '/X_train.npy'


    # FS: turn this off for now as it is causing code to fall over
    if os.path.exists(path_to_npy):
        data_set.load_processed_data()
    else:
        data_set.load_new_data()

    return data_set


def one_train(path, data_set, model_function, parameters):
    os.makedirs(path)
    os.makedirs(path + '/models')
    os.makedirs(path + '/plots')
    os.makedirs(path + '/confusion_matrix')
    os.makedirs(path + '/conf_matrix_csv')
    os.makedirs(path + '/conf_matrix_details')

    batch_size = 0
    epochs = 0

    params = list()
    for p in parameters:
        message = str(p) + ' : '
        value = input(message)

        if p == 'batch_size':
            batch_size = int(value)
        elif p == 'epochs':
            epochs = int(value)
        else:
            params.append(str(value))

    train_model = model_function(np.shape(data_set.X_train), params)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')

    params.append(batch_size)
    params.append(epochs)
    history = TrainingHistoryPlot(path, data_set, params)

    print("Training...")
    train_model.fit(data_set.X_train, data_set.Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(data_set.X_valid, data_set.Y_valid),
                    callbacks=[early_stopper, history])
    print("Done!")

    score = train_model.evaluate(data_set.X_valid, data_set.Y_valid, verbose=0)

    file = open(path + '/result.txt', 'w')
    file.write('Test loss : ' + str(score[0]) + '\n')
    file.write('Test acc : ' + str(score[1]) + '\n')
    file.close()


def main():
    if len(sys.argv) > 1:
        run_num = int(sys.argv[1])
    else:
        run_num = 0
    #run_num = int(run_num)
    print(run_num)
    random.seed(run_num)
    np.random.seed(run_num)
    #MODEL_NAME = input("Please introduce model name (dgn, conv3d, lstm_bucketing, lstm_sliding): ")
    #GA = input("Do you want to use genetic algorithm for your model? (yes/no): ")
    #GS = input("Do you want to use grid search for your model? (yes/no): ")
    MODEL_NAME = 'lstm_sliding'
    GA = 'yes'
    GS = 'no'
    # NAIVE - RANDOM FILL W/ ACCURACY (not very good but matched GitHub code most closely)
    #mo_type = 'naive-rand'
    # NAIVE - TOURNAMENT SELECTION W/ OBJECTIVE AGGREGATE (as described in the NeuroEvolutionary paper)
    # - *** This is used in experiments ***
    #mo_type = 'naive-tournament-select'
    # NSGA-II
    mo_type = 'nsga-ii'
    # moead
    #mo_type = 'moead'
    # moead gra
    #mo_type = 'moead_gra'

    time_str = time.strftime("%Y-%m-%d_%H %M")
    path = PATH_SAVE_FIG + str(time_str) + '_' + str(run_num)

    # Can potentially have more models here
    if(MODEL_NAME == 'lstm_sliding'):
        parameters = PARAMETERS_LSTM
        model_function = model.lstm_model
        #model_function = model.lstm_autoencoder_model
        #model_function = model.lstm_test

    print('\nLoad dataset...')
    data_set = load_data(MODEL_NAME)
    print('Done! \n')

    #sys.exit()

    # ************* SUPER SCALER ************** #

    print("****** Scaling X and Y pos. independantly *******")
    np.set_printoptions(threshold=np.inf)
    #def denormalize(array, max, min):
    #    return array*(max - min) + min

    #Y_train_rescale = denormalize(data_set.Y_train, data_set.ymax, data_set.ymin)
    Y_train_x_pos = data_set.Y_train[:, ::2]
    #Y_train_y_pos = data_set.Y_train[:, 1::2]

    def normalize(array, min, max):
        return (array - min)/ (max - min)

    def normalize_x(array, min, max):
        return (array[:, ::2] - min)/ (max - min)

    def normalize_y(array, min, max):
        return (array[:, 1::2] - min)/ (max - min)

    def y_subtract(array):
        array_copy = copy.deepcopy(array)
        for i in range(1, int(array.shape[1]/2) ):
            #print(i)
            array[:, i*2 + 1] = array_copy[:, i*2 + 1] - array_copy[:, (i*2 - 1)]
            array[:, i*2] = array_copy[:, i*2] - array_copy[:, (i*2 - 2)]
        return array

    # If jump in Y position does not make sense impute the previous or next positional y value
    # [i.e if value is high due to acc this will remain high and vica versa]
    def impute_next_pos(array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] > 10 and j < (array.shape[1]-1):
                    array[i, j] = array[i, j + 2]
                    array[i, j - 1] = array[i, j + 1]
                elif array[i, j] > 10 and j >= (array.shape[1]-1) :
                    array[i, j] = array[i, j - 2]
                    array[i, j - 1] = array[i, j - 3]
        return array



    data_set.Y_train = y_subtract(data_set.Y_train)
    data_set.Y_test = y_subtract(data_set.Y_test)
    data_set.Y_valid = y_subtract(data_set.Y_valid)

    data_set.Y_train = impute_next_pos(data_set.Y_train)
    data_set.Y_test = impute_next_pos(data_set.Y_test)
    data_set.Y_valid = impute_next_pos(data_set.Y_valid)

    data_set.Y_full = y_subtract(data_set.Y_full)
    data_set.Y_full = impute_next_pos(data_set.Y_full)

    data_set.superscaler_x_min = np.amin(data_set.Y_full[:, ::2])
    data_set.superscaler_x_max = np.amax(data_set.Y_full[:, ::2])
    data_set.superscaler_y_min = np.amin(data_set.Y_full[:, 1::2])
    data_set.superscaler_y_max = np.amax(data_set.Y_full[:, 1::2])


    #data_set.Y_train  = normalize(data_set.Y_train, data_set.superscaler_x_min, data_set.superscaler_x_max)
    #data_set.Y_test = normalize(data_set.Y_test, data_set.superscaler_x_min, data_set.superscaler_x_max)
    #data_set.Y_valid = normalize(data_set.Y_valid, data_set.superscaler_x_min, data_set.superscaler_x_max)

    data_set.Y_train[:, ::2]  = normalize_x(data_set.Y_train, data_set.superscaler_x_min, data_set.superscaler_x_max)
    data_set.Y_test[:, ::2] = normalize_x(data_set.Y_test, data_set.superscaler_x_min, data_set.superscaler_x_max)
    data_set.Y_valid[:, ::2] = normalize_x(data_set.Y_valid, data_set.superscaler_x_min, data_set.superscaler_x_max)

    data_set.Y_train[:, 1::2]  = normalize_y(data_set.Y_train, data_set.superscaler_y_min, data_set.superscaler_y_max)
    data_set.Y_test[:, 1::2]  = normalize_y(data_set.Y_test, data_set.superscaler_y_min, data_set.superscaler_y_max)
    data_set.Y_valid[:, 1::2]  = normalize_y(data_set.Y_valid, data_set.superscaler_y_min, data_set.superscaler_y_max)

    #data_set.Y_train[:, 1::2]

    #print(data_set.superscaler_x_min)
    #print(data_set.superscaler_x_max)

    #np.set_printoptions(precision=3)
    #for i in range(len(data_set.Y_valid)):
    #    print(str(data_set.Y_valid[i]) + '\r')
    #    print("\n")

    #sys.exit()

    # This has not really been tested, using model.lstm_test when testing
    if GA == 'no' and GS == 'no':
        one_train(path, data_set, model_function, parameters)

    if GA == 'yes':
        optim = GeneticAlgorithm(path, parameters, model_function, data_set, run_num)
        optim.run(mo_type)

    if GS == 'yes':
        optim = GridSearch(path, parameters, model_function, data_set, run_num)
        optim.run(mo_type)

    # Print out at end for ease, this is hardcoded at the moment in load_model_analysis
    print(data_set.superscaler_x_min)
    print(data_set.superscaler_x_max)
    print(data_set.superscaler_y_min)
    print(data_set.superscaler_y_max)

if __name__ == '__main__':
    #var = str(int(sys.argv[1]) % 2)
    #with tf.device('/device:GPU:'+var):
    main()
