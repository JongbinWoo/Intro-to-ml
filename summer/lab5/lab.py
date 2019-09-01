import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

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
        for i in range(data.shape[0]):
            if (k == label[i]):
                j += 1
                x1 += data[i,0]
                x2 += data[i,1]
        M[k,0] = x1 / j
        M[k,1] = x2 / j
    return M

data_name = "./data_kmeans.txt"
columns = ['x1', 'x2']
data = pd.read_csv(data_name, names=columns, sep=' ')
data = data.as_matrix()
plt.plot(data[:,0],data[:,1],'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.plot(data[:,0],data[:,1],'ro')
plt.plot(data[:,0],data[:,1],'ro')
plt.plot(data[label==0,0],data[label==0,1],'ro')
plt.plot(data[label==1,0],data[label==1,1],'bo')
plt.plot(data[label==2,0],data[label==2,1],'yo')
plt.plot(M[:,0], M[:,1], 'gx', label='Centroids')
test_data=np.array([[0.8, 1.2], [7., 5.],[1, 4]])
test_distance=distance(M, test_data)
test_label=labeling(test_distance)
plt.plot(test_data[test_label==0, 0], test_data[test_label==0, 1], "rx", label='Test')
plt.plot(test_data[test_label==1, 0], test_data[test_label==1, 1], "bx", label='Test')
plt.plot(test_data[test_label==2, 0], test_data[test_label==2, 1], "yx", label='Test')
plt.legend()
plt.show()
