import pandas as pd
import numpy as np
from os import walk
import math
import matplotlib.pyplot as plt
import sys

#TODO need to arrange code in functions and classes

class LoadTraj:
    def getTraj():
        df = pd.read_csv("images/state_buf.txt", sep='\t', header=None)
        print(df.head())
        sys.exit()

        df_training = pd.DataFrame()
        df_testing = pd.DataFrame()
        df_validation = pd.DataFrame()

        folder_list = ["training", "testing", "validation"]

        for folder in folder_list:
            #print(folder)
            mypath = "images_splited/"+folder+"/image"

            # Walk the path
            filenames = next(walk(mypath), (None, None, []))[2]
            index_list = []

            seq_len = 3

            for filename in filenames:
                end_of_timestamp_index = filename.find('_')
                timestamp = int(filename[:end_of_timestamp_index])
                index_list.append(timestamp)

            if folder == "training":
                df_training = df.loc[df.index[index_list]]
                print("Training df shape:"+str(df_training.shape))
                p = df_training.to_numpy()
                l1 = l1_objective(p, seq_len)
                l2 = l2_objective(p, seq_len)
                l3 = l3_objective(p, seq_len)
                Obj_train = np.column_stack((l1, l2, l3))
                Y_train = np.array(trajectory(p, seq_len))
                print("Got here no?")
                print(Y_train)
            elif folder == "testing":
                df_testing = df.loc[df.index[index_list]]
                print("Testing df shape:"+str(df_testing.shape))
                p = df_testing.to_numpy()
                l1 = l1_objective(p, seq_len)
                l2 = l2_objective(p, seq_len)
                l3 = l3_objective(p, seq_len)
                Obj_test = np.column_stack((l1, l2, l3))
                Y_test = np.array( trajectory(p, seq_len))
            elif folder == "validation":
                df_validation = df.loc[df.index[index_list]]
                print("Validation df shape:"+str(df_validation.shape))
                p = df_validation.to_numpy()
                l1 = l1_objective(p, seq_len)
                l2 = l2_objective(p, seq_len)
                l3 = l3_objective(p, seq_len)
                Obj_validation = np.column_stack((l1, l2, l3))
                Y_validation = np.array(trajectory(p, seq_len))
            else:
                print("folder does not exist in list of folders, required list: " + folder)
                sys.exit()

        return Y_train, Y_test, Y_validation


def trajectory(p, seq_len):
    traj = []
    for i in range(math.floor(len(p)/seq_len)-1):
        j = seq_len * i
        coord = []
        x0 = p[j][0]
        y0 = p[j][1]
        j = seq_len * i
        # The image filename will be dropped later in get_input_sequences()
        #coord.append
        for k in range(1, seq_len):
            #print(k)
            xk = p[j+k][0]
            yk = p[j+k][1]
            # These may be neg
            x = xk - x0
            y = yk - y0
            coord.append(x)
            coord.append(y)
        traj.append(coord)
    return traj


# Eqn 3. NeuroTrajectory distance-based feedback
def l1_objective(p, seq_len):
    """
    Calculate Eqn 3. NeuroTrajectory distance-based feedback
    :params: Numpy array of input data
    :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
    """
    l1 = []
    for i in range(math.floor(len(p)/seq_len)-1):
        j = seq_len * i
        temp=0.0
        for k in range(seq_len):
            x2 = p[j+k][0]
            x1 = p[j+(seq_len-1)][0]
            y2 = p[j+k][1]
            y1 = p[j+(seq_len-1)][1]
            temp += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        l1.append(temp)
    return l1

def l2_objective(p, seq_len):
    """
    Calculate Eqn 4. NeuroTrajectory lateral velocity
    :params: Numpy array of input data
    :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
    """
    l2 = []
    # Calculate velocity as rate of change in position across fixed sequence length
    for i in range(math.floor(len(p)/seq_len)-1):
        j = seq_len * i
        vd = 0.0
        for k in range(seq_len):
            # diff between each sequence
            vd += np.abs(p[j+k+1][0] - p[j+k][0])/seq_len
        l2.append(vd)
    return l2

def l3_objective(p, seq_len):
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    l3 = []
    # Calculate velocity as rate of change in position across fixed sequence length
    for i in range(math.floor(len(p)/seq_len)-1):
        j = seq_len * i
        vf = 0.0
        for k in range(seq_len):
            # diff between each sequence
            vf += np.abs(p[j+k+1][1] - p[j+k][1])/seq_len
        l3.append(vf)
    return l3


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
