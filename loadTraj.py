import pandas as pd
import numpy as np
from os import walk
import math

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

l1 = []

print(math.floor(len(p)/seq_len))
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

print(l1)
print(len(l1))
