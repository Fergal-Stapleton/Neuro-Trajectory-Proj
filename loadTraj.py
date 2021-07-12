import pandas as pd
import numpy as np
from os import walk
import math
import matplotlib.pyplot as plt

#TODO need to arrange code in functions and classes

def main():
    df = pd.read_csv("images/replay.txt", sep=',', header=None)
    print(df.head())

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

        for filename in filenames:
            end_of_timestamp_index = filename.find('_')
            timestamp = int(filename[:end_of_timestamp_index])
            index_list.append(timestamp)

        if folder == "training":
            df_training = df.loc[df.index[index_list]]
        elif folder == "testing":
            df_testing = df.loc[df.index[index_list]]
        elif folder == "validation":
            df_validation = df.loc[df.index[index_list]]
        else:
            print("folder does not exist in list of folders, required list: " + folder)
            sys.exit()

    print("Training df shape:"+str(df_training.shape))
    print("Testing df shape:"+str(df_testing.shape))
    print("Validation df shape:"+str(df_validation.shape))

    p = df_training.to_numpy()
    print(p)
    seq_len = 3

    # Objectives for our MO
    l = [l1, l2, l3]

    #plt.scatter(l1,l3)
    #plt.show()

    # TODO add l2 here
    Y_train = np.column_stack((l1, l3))

    # Can do a simple test to check dominance function works
    # When properly implemented Will be doing this with predicted Y
    non_dom = is_pareto_efficient_simple(Y_train)
    print(non_dom)

# Eqn 3. NeuroTrajectory distance-based feedback
def l1_objective():
    """
    Calculate Eqn 3. NeuroTrajectory distance-based feedback
    :params: Numpy array of input data
    :return: list, each element is the distance-based feedback corresponding to the ith sequence of OGs
    """
    l1 = []
    for i in range(0,math.floor(len(p)/seq_len)-1):
        j = seq_len * i
        temp=0.0
        for k in range(0, seq_len - 1):
            x2 = p[j+k][0]
            x1 = p[j+k+(seq_len-1)][0]
            y2 = p[j+k][1]
            y1 = p[j+k+(seq_len-1)][1]
            temp += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        l1.append(temp)
    return l1

def l2_objective():
    """
    Calculate Eqn 4. NeuroTrajectory lateral velocity
    :params: Numpy array of input data
    :return: list, each element is the lateral velocity corresponding to the ith sequence of OGs
    """
    l2 = []
    # Calculate velocity as rate of change in position across fixed sequence length
    for i in range(0,math.floor(len(p)/seq_len)-1):
        vd = 0.0
        for k in range(0, seq_len - 2):
            # diff between each sequence
            vd += p[i+k+1][0] - p[i+k][0]
        vd = vd/seq_len
        l2.append(vd)
    return l2

def l3_objective():
    """
    Calculate Eqn 5. NeuroTrajectory longtitudinal velocity
    :params: Numpy array of input data
    :return: list, each element is the longtitudinal velocity corresponding to the ith sequence of OGs
    """
    l3 = []
    # Calculate velocity as rate of change in position across fixed sequence length
    for i in range(0,math.floor(len(p)/seq_len)-1):
        vf = 0.0
        for k in range(0, seq_len - 2):
            # diff between each sequence
            vf += p[i+k+1][1] - p[i+k][1]
        vf = vf/seq_len
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

main()
