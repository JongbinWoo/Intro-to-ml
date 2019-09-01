import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

data_name = "./grade_students.csv"
columns = ['x1', 'x2','x3', 'x4','x5', 'x6']
data = pd.read_csv(data_name, names=columns, sep=',')
#x_1 = np.asarray(data['x1']).reshape(-1,1)
#x_2 = np.asarray(data['x2']).reshape(-1,1)
#x_3 = np.asarray(data['x3']).reshape(-1,1)
#x_4 = np.asarray(data['x4']).reshape(-1,1)
#x_5 = np.asarray(data['x5']).reshape(-1,1)
#x_6 = np.asarray(data['x6']).reshape(-1,1)
#data_t = np.hstack((x_1,x_2,x_3,x_4,x_5,x_6))
data_m = data.as_matrix()

data = data_m[1:,:]
data = np.asfarray(data,float)
data
M = np.random.random_sample(size=(3, data.shape[1]))

M
def normalize(A):
    B = (A - np.min(A, axis=0)) / (np.max(A, axis=0) -np.min(A, axis=0))
    return B

def un_normalize(A,M):
    M = (np.max(A, axis=0) -np.min(A, axis=0)) * M + np.min(A, axis=0)
    return M
n_data = normalize(data)
n_data

def distance(M, data):
    A = np.zeros((data.shape[0],M.shape[0]), dtype=np.float)
    for i in range(data.shape[0]):
        for j in range(M.shape[0]):
            A[i,j] = np.linalg.norm(data[i] - M[j])
            #B = np.argmin(A, axis=1)
    return A

def labeling(A):
    A = np.argmin(A, axis=1)
    return A

def update(M,data,label):
    for k in range(M.shape[0]):
        j = 0
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        x5 = 0
        x6 = 0
        for i in range(data.shape[0]):
            if (k == label[i]):
                j += 1
                x1 += data[i,0]
                x2 += data[i,1]
                x3 += data[i,2]
                x4 += data[i,3]
                x5 += data[i,4]
                x6 += data[i,5]
        if j==0: j=0.0001
        M[k,0] = x1 / j
        M[k,1] = x2 / j
        M[k,2] = x3 / j
        M[k,3] = x4 / j
        M[k,4] = x5 / j
        M[k,5] = x6 / j
    return M

for i in range(100):
    A = distance(M,n_data)
    label = labeling(A)
    M = update(M,n_data,label)
M
M = un_normalize(data,M)
data_m[0]
Group1 = M[0]
print(Group1)
Group2 = M[1]
print(Group2)
Group3 = M[2]
print(Group3)
