#A) Reducing the dimension of some synthetic data
#1. Download from the course site the 2D data stored in data pca.txt file.
#2. Implement the PCA algorithm from the formulas seen in class.
#3. Indicate the principal axes of the data.
#4. Test your model with some new data.
#5. Plot both training and test results in a 2D graph.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
data_name = "./data_pca.txt"
columns = ['x1', 'x2']
data = pd.read_csv(data_name, names=columns, sep=' ')
x_1 = np.asarray(data['x1']).reshape(-1,1)
x_2 = np.asarray(data['x2']).reshape(-1,1)
X = np.hstack((x_1,x_2))

plt.figure(5)
plt.plot(x_1,x_2,'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')

M = np.sum(X, axis=0) / X.shape[0]
X_ = X - M
X_
plt.figure(5)
plt.plot(X_[:,0],X_[:,1],'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
def compute_sum(X):
    s=0
    for i in range(X.shape[0]):
        s += np.multiply(X[i],X[i].reshape((2,1)))
    s /=2
    return s

sum = compute_sum(X_)
sum
lamb, U = np.linalg.eig(sum)

U
lamb

pc_index = np.argmax(lamb)
pc = U[:,pc_index]
pc
plt.plot([0,pc[0]],[0, pc[1]])
plt.show()

def projection(data,u):
#    P = np.zeros_like(data, dtype=float)
#    for i in range(data.shape[0]):
#        P[i] = np.dot(u, data[i]) / np.linalg.norm(u)
    P = np.dot(data,u)
    return P



P = projection(X_+M,pc)
X_
P
plt.plot(P,np.zeros_like(P),'ro')
plt.xlabel('x_1')

#plt.plot([0,pc[0]],[0, pc[1]])
plt.show()
plt.plot(X_[:,0],X_[:,1],'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
