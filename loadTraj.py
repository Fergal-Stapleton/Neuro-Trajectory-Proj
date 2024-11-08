import pandas as pd
import numpy as np
from os import walk
import math
import matplotlib.pyplot as plt
import sys
from natsort import index_natsorted

#TODO need to arrange code in functions and classes

class LoadTraj:

    def getTraj(number_of_classes, slide):
        df = pd.read_csv("images/state_buf.txt", sep='\t')
        #print(df.shape[0])
        #print(df.head())
        #sys.exit()

        df_training = pd.DataFrame()
        df_testing = pd.DataFrame()
        df_validation = pd.DataFrame()

        folder_list = ["training", "testing", "validation"]

        for folder in folder_list:
            df = pd.read_csv("images/state_buf.txt", sep='\t')
            #print(folder)
            mypath = "images_splited/"+folder+"/image"

            # Walk the path
            filenames = next(walk(mypath), (None, None, []))[2]
            index_list = []
            #print(df.index[344])
            seq_len = int((number_of_classes + 2) /2)
            df['Index'] = df['ImageName'].str.split('_', 0).str[0]
            #print(df['Index'])

            for filename in filenames:
                end_of_timestamp_index = filename.find('_')
                timestamp = int(filename[:end_of_timestamp_index])
                # TODO need a way to find lowest id in folder
                index_list.append(str(timestamp))

            if folder == "training":
                #df_training = df.loc[df.index[index_list]]
                df_training = df[df['Index'].isin(index_list)]
                print("Training df shape:"+str(df_training.shape))
                df_training = df_training.sort_values(by="ImageName", key=lambda x: np.argsort(index_natsorted(df_training["ImageName"])))
                p = df_training.to_numpy()
                #l1 = l1_objective(p, seq_len)
                #l2 = l2_objective(p, seq_len)
                #l3 = l3_objective(p, seq_len)
                #Obj_train = np.column_stack((l1, l2, l3))
                if slide == True:
                    Y_train = trajectory_plus_one(p, seq_len)
                else:
                    Y_train = trajectory_seq_len(p, seq_len)
            elif folder == "testing":
                #df_testing = df.loc[df.index[index_list]]
                df_testing = df[df['Index'].isin(index_list)]
                print(df_testing)
                print("Testing df shape:"+str(df_testing.shape))
                df_testing = df_testing.sort_values(by="ImageName", key=lambda x: np.argsort(index_natsorted(df_testing["ImageName"])))
                p = df_testing.to_numpy()
                #l1 = l1_objective(p, seq_len)
                #l2 = l2_objective(p, seq_len)
                #l3 = l3_objective(p, seq_len)
                #Obj_test = np.column_stack((l1, l2, l3))
                if slide == True:
                    Y_test = trajectory_plus_one(p, seq_len)
                else:
                    Y_test = trajectory_seq_len(p, seq_len)
            elif folder == "validation":
                #df_validation = df.loc[df.index[index_list]]
                df_validation = df[df['Index'].isin(index_list)]
                #print("Validation df shape:"+str(df_validation.shape))
                df_validation = df_validation.sort_values(by="ImageName", key=lambda x: np.argsort(index_natsorted(df_validation["ImageName"])))
                p = df_validation.to_numpy()
                #l1 = l1_objective(p, seq_len)
                #l2 = l2_objective(p, seq_len)
                #l3 = l3_objective(p, seq_len)
                #Obj_validation = np.column_stack((l1, l2, l3))
                if slide == True:
                    Y_validation = trajectory_plus_one(p, seq_len)
                else:
                    Y_validation = trajectory_seq_len(p, seq_len)
            else:
                print("folder does not exist in list of folders, required list: " + folder)
                sys.exit()

        return Y_train, Y_test, Y_validation


def trajectory_seq_len(p, seq_len):
    traj = []

    for i in range(math.floor(len(p)/seq_len)):
        j = seq_len * i
        coord = []
        vel_long_tmp = []
        vel_angl_tmp = []
        # Trajectory designed so ego vehicle is always at [0, 0]
        x0 = p[j][0]
        y0 = p[j][1]
        j = seq_len * i
        # The image filename will be dropped later in get_input_sequences()
        #coord.append(str(p[j][5]).strip('_image.png'))
        coord.append(p[j][6])

        #print(p[j][-1])
        for k in range(1, seq_len):
            # previous index is -1 of current

            xk = p[j+k][0]
            yk = p[j+k][1]
            # These may be neg
            x = xk - x0
            y = yk - y0
            coord.append(x)
            coord.append(y)
            #x0 = xk
            #y0 = yk
        traj.append(coord)
    return np.array(traj)

def trajectory_plus_one(p, seq_len):
    traj = []
    for i in range(len(p)- seq_len):
        coord = []
        # Trajectory designed so ego vehicle is always at [0, 0]
        x0 = p[i][0]
        y0 = p[i][1]
        # The image filename will be dropped later in get_input_sequences()
        #coord.append(str(p[j][5]).strip('_image.png'))
        coord.append(p[i][6])
        #print(p[j][5])
        for k in range(1, seq_len):
            # previous index is -1 of current

            #print(k)
            xk = p[i+k][0]
            yk = p[i+k][1]
            x = xk - x0
            y = yk - y0
            coord.append(x)
            coord.append(y)
        traj.append(coord)
    return np.array(traj)

#
# # Eqn 3. NeuroTrajectory distance-based feedback
# def l1_objective(p, seq_len):
#     """
#     Calculate Eqn 3. NeuroTrajectory distance-based feedback
#     :params: Numpy array of input data
#     :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
#     """
#     l1 = []
#     for i in range(math.floor(len(p)/seq_len)):
#         j = seq_len * i
#         temp=0.0
#         for k in range(seq_len):
#             x2 = p[j+k][0]
#             x1 = p[j+(seq_len-1)][0]
#             y2 = p[j+k][1]
#             y1 = p[j+(seq_len-1)][1]
#             temp += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         l1.append(temp)
#     return l1
#
# def l2_objective(p, seq_len):
#     """
#     Calculate Eqn 4. NeuroTrajectory lateral velocity
#     :params: Numpy array of input data
#     :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
#     """
#     l2 = []
#     # Calculate velocity as rate of change in position across fixed sequence length
#     for i in range(math.floor(len(p)/seq_len)):
#         j = seq_len * i
#         vd = 0.0
#         for k in range(seq_len):
#             # diff between each sequence
#             vd += np.abs(p[j+k+1][0] - p[j+k][0])/seq_len
#         l2.append(vd)
#     return l2
#
# def l3_objective(p, seq_len):
#     """
#     Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
#     :params: Numpy array of input data
#     :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
#     """
#     l3 = []
#     # Calculate velocity as rate of change in position across fixed sequence length
#     for i in range(math.floor(len(p)/seq_len)):
#         j = seq_len * i
#         vf = 0.0
#         for k in range(seq_len):
#             # diff between each sequence
#             vf += np.abs(p[j+k+1][1] - p[j+k][1])/seq_len
#         l3.append(vf)
#     return l3


# Pareto dominance code taken from:
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :params: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient
