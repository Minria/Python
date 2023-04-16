import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

data_path_mat = 'D:\wangfuming\Desktop\CNND/input\AVIRIS-I.mat'
data_path_csv = 'D:\wangfuming\Desktop\CNND/result\AVIRIS-I.csv'
gt = sio.loadmat(data_path_mat)['gt']
df = pd.read_csv(data_path_csv)
df = df.values
m = min(gt.shape[0],df.shape[0])
n = min(gt.shape[1],df.shape[1])
TP = 0
FP = 0
TN = 0
FN = 0
rate = 0.5
for i in range(m):
    for j in range(n):
        if df[i][j] >= rate and gt[i][j] == 1:
            TP+=1
        if df[i][j] >= rate and gt[i][j] == 0:
            FP+=1
        if df[i][j] < rate and gt[i][j] == 1:
            FN+=1
        if df[i][j] < rate and gt[i][j] == 0:
            TN+=1


print(TP)
print(FP)
print(FN)
print(TN)
print(TP*1.0/(TP+FN))
print(FP*1.0/(TN+FP))
