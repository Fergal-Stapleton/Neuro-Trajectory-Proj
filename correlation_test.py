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
from scipy import stats
import matplotlib.pyplot as plt
import pandas
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

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


# Eqn 3. NeuroTrajectory distance-based feedback
def l1_objective(p, image_sequence_length):
    """
    Calculate Eqn 3. NeuroTrajectory distance-based feedback
    :params: Numpy array of input data
    :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
    """
    l1 = 0.0
    dest_list = []
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        if flag == 1:
            # Y_test [0.1, 5.5, 0.3, 10.7]
            # Since our start point is implicitly always [0.0, 0.0]
            # (image_sequence_length-1)*2 -1 last coordinate
            x_tau = (image_sequence_length-1)*2 -1
            # (image_sequence_length-1)*2 -1 Second last coordinate
            y_tau = (image_sequence_length-1)*2 -2
            # P_ego <t+0> - P_dest <t+tau>
            temp += np.sqrt((p[i][x_tau] - 0.0)**2 + (p[i][y_tau] - 0.0)**2)
            flag = 0
        # these go up by 2,  P ego <t+i> - P dest <t+tau> , where i > 0
        for j in range(2, x_tau):
            P_dest_x = p[i][x_tau] # starts at 3
            P_ego_x = p[i][j - 1] # starts at 1
            P_dest_y = p[i][y_tau] # starts at 2
            P_ego_y = p[i][j - 2] # starts at 0
            temp += np.sqrt((P_ego_x  - P_dest_x)**2 + (P_ego_y - P_dest_y)**2)
        dest_list.append(temp/image_sequence_length)
    l1 = dest_list
    return l1

# Eqn 3. NeuroTrajectory distance-based feedback
def l1_objective_original(p, image_sequence_length):
    """
    Calculate Eqn 3. NeuroTrajectory distance-based feedback
    :params: Numpy array of input data
    :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
    """
    l1 = 0.0
    dest_list = []
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        if flag == 1:
            # Y_test [0.1, 5.5, 0.3, 10.7]
            # Since our start point is implicitly always [0.0, 0.0]
            # (image_sequence_length-1)*2 -1 last coordinate
            x_tau = (image_sequence_length-1)*2 -1
            # (image_sequence_length-1)*2 -1 Second last coordinate
            y_tau = (image_sequence_length-1)*2 -2
            # P_ego <t+0> - P_dest <t+tau>
            temp += np.sqrt((p[i][x_tau] - 0.0)**2 + (p[i][y_tau] - 0.0)**2)
            flag = 0
        # these go up by 2,  P ego <t+i> - P dest <t+tau> , where i > 0
        for j in range(2, x_tau):
            P_dest_x = p[i][x_tau] # starts at 3
            P_ego_x = p[i][j - 1] # starts at 1
            P_dest_y = p[i][y_tau] # starts at 2
            P_ego_y = p[i][j - 2] # starts at 0
            temp += np.sqrt((P_ego_x  - P_dest_x)**2 + (P_ego_y - P_dest_y)**2)
        dest_list.append(temp/image_sequence_length)
    l1 = dest_list
    return l1

def l2_objective(p, t_delta, slide, image_sequence_length):
    """
    Calculate Eqn 4. NeuroTrajectory max angular velocity
    :params: Numpy array of input data
    :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l2 = 0.0
    vd = []
    vector_len = (image_sequence_length-3)*2+1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        # np.arctan2(b[i+1][0] - b[i][0], b[i+1][1] - b[i][1]) - np.arctan2(b[i][0] - b[i-1][0], b[i][1] - b[i-1][1]))/0.2
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            #
            temp += np.abs((np.arctan2(p[i][2] - p[i][0], p[i][3] - p[i][1]) - np.arctan2(p[i][0] - 0.0, p[i][1] - 0.0))/t_delta)
        if image_sequence_length > 3:
            for j in range(2,vector_len):
                temp += np.abs((np.arctan2(p[i][j+2] - p[i][j], p[i][j+3] - p[i][j+1]) - np.arctan2(p[i][j] - p[i][j-2], p[i][j+1] - p[i][j-1]))/t_delta)
        vd.append(temp)
    l2 = vd
    return l2


def l3_objective(p, t_delta, max_vel, slide, image_sequence_length):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    # Calculate velocity as rate of change in position across fixed sequence length
    l3 = 0.0
    vf = []
    vector_len = (image_sequence_length-1)*2 -1
    t_idx = -1
    idx = -1
    for i in range(p.shape[0]):
        temp = 0.0
        flag = 1
        t_idx += 1
        if slide == True:
            t_idx = idx + 1
        if flag == 1:
            # Since our start point is implicitly always [0.0, 0.0]
            idx = t_idx
            temp += np.abs(max_vel - ((p[i][1] - 0.0)/t_delta))
            flag = 0
        # these go up by 2
        for j in range(2,vector_len):
            idx += 1
            temp += np.abs(max_vel - ((p[i][j+1] - p[i][j - 1])/t_delta))
        vf.append(temp/image_sequence_length)
    #print(vf)
    l3 = vf
    return l3

def main():

    MODEL_NAME = 'lstm_sliding'
    GA = 'yes'
    GS = 'no'

    print('\nLoad dataset...')
    data_set = load_data(MODEL_NAME)
    print('Done! \n')


    # data_set.Y_train
    image_sequence_length = int((data_set.number_of_classes + 2) /2)
    print(data_set.Y_train.shape[0])

    def impute_next_pos(array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] > 10 and j < (array.shape[1]-1):
                    array[i, j] = array[i, j + 2]
                elif array[i, j] > 10 and j >= (array.shape[1]-1) :
                    array[i, j] = array[i, j - 2]
        return array

    #data_set.Y_train = impute_next_pos(data_set.Y_train)

    # This needs to be found out from GridSim or by diff'n timestamps of images
    t_delta = 0.2
    max_vel = 32.5 # 130 / 5 = 32.5
    l1 = l1_objective(data_set.Y_train, image_sequence_length)
    l2 = l2_objective(data_set.Y_train, t_delta, data_set.slide, image_sequence_length)
    l3 = l3_objective(data_set.Y_train, t_delta, max_vel, data_set.slide, image_sequence_length)

    # To show that values are not normal
    #pyplot.hist(l1)
    #pyplot.show()



    l1_l2 = stats.spearmanr(l1, l2, axis=None)
    l1_l3 = stats.spearmanr(l1, l3, axis=None)
    l2_l3 = stats.spearmanr(l2, l3, axis=None)

    plt.scatter(l1, l2)
    plt.xlabel('l1')
    plt.ylabel('l2')
    plt.title('Spearman rank-order correlation coefficient: '+ str(format(l1_l2[0] , ".3f")))
    plt.savefig('plots/spear_corr_l1_l2.png')
    plt.clf()

    plt.scatter(l1, l3)
    plt.xlabel('l1')
    plt.ylabel('l3')
    plt.title('Spearman rank-order correlation coefficient: '+ str(format(l1_l3[0] , ".3f")))
    plt.savefig('plots/spear_corr_l1_l3.png')
    plt.clf()

    plt.scatter(l2, l3)
    plt.xlabel('l2')
    plt.ylabel('l3')
    plt.title('Spearman rank-order correlation coefficient: '+ str(format(l2_l3[0] , ".3f")))
    plt.savefig('plots/spear_corr_l2_l3.png')
    plt.clf()

    print(l1_l2)
    print(l1_l3)
    print(l2_l3)

    parallel_coordinates(np.concatenate([l1_l2, l1_l3, l2_l3]))

if __name__ == '__main__':
    main()
