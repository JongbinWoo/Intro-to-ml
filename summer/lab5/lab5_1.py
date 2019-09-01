#A) Clustering some synthetic data
#1. Download from the course site the 2D data stored in data kmeans.txt file.
#2. Cluster them using the K-means algorithm using the formulas seen in class.
#3. Test your model with some new data.
#4. Plot both training and test results in a 2D graph.
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

data_name = "./data_kmeans.txt"
columns = ['x1', 'x2']
data = pd.read_csv(data_name, names=columns, sep=' ')
x_1 = np.asarray(data['x1']).reshape(-1,1)
x_2 = np.asarray(data['x2']).reshape(-1,1)
data = np.hstack((x_1,x_2))

plt.figure(3)
plt.plot(x_1,x_2,'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')

M = np.random.randint(8, size=(3, 2))
M = M.astype(float)

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

A = distance(M,data)
A = labeling(A)

def update(M,data,label):
    for k in range(M.shape[0]):
        j = 0
        x1 = 0
        x2 = 0
        for i in range(data.shape[0]):
            if (k == label[i]):
                j += 1
                x1 += data[i,0]
                x2 += data[i,1]

        M[k,0] = x1 / j
        M[k,1] = x2 / j
    return M

for i in range(10):
    A = distance(M,data)
    label = labeling(A)
    M = update(M,data,label)

result = np.concatenate([data,label.reshape(-1,1)],axis=1)

test = np.
