import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def Initialization(N, K, J):
    V = np.random.randn(N,K)
    W = np.random.randn(K+1,J)
    return V,W


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def FP(V,W,X):
    X__ = np.matmul(X,V)
    F = sigmoid(X__)
    F_ = np.concatenate([a,F], axis =1)
    F__ = np.matmul(F_, W)
    G = sigmoid(F__)
    #G = np.eye(2)[np.argmax(G,axis=1)]
    return G,F,F_

def BP(V,W,G,Y,F_,F,lr ):
    for k in range(0,W.shape[0]):
        for j in range(0,W.shape[1]):
            value =0
            for i in range(1,len(Y)):
                value += (G[i][j] - Y[i][j])*G[i][j]*(1-G[i][j])*F_[i][k]
            W[k][j] = W[k][j] - lr * value


    for n in range(0,V.shape[0]):
        for k in range(0,V.shape[1]):
            value =0
            for i in range(1,len(Y)):
                for j in range(1,Y.shape[1]):
                    value += (G[i][j] - Y[i][j])*G[i][j]*(1-G[i][j])*W[k][j]*F[i][k]*(1-F[i][k])*X[i][n]
            V[n][k] = V[n][k] - lr * value

    return V,W

data_name = "./data_FFNN.txt"
columns = ['x1', 'x2', 'y']
data = pd.read_csv(data_name, names=columns, sep=' ')
x_1 = np.asarray(data['x1'])
x1 = np.expand_dims(x_1, axis=1)
x_2 = np.asarray(data['x2'])
x2 = np.expand_dims(x_1, axis=1)
y = np.asarray(data['y'])
Y = np.eye(2)[y]
a = np.ones_like(x1)
X = np.concatenate([a, x1, x2], axis=1)

N = X.shape[1]
K = 5 #hidden neuron
J = 2 # of classes

lr = 0.001

V,W = Initialization(N,K,J)
print(V)
print(W)



for i in range(10000):
    Return =  FP(V,W,X)
    G = Return[0]
    F = Return[1]
    F_ = Return[2]

    Return = BP(V,W,G,Y,F_,F ,lr)
    V = Return[0]
    W = Return[1]
print(V)
Return =  FP(V,W,X)
G = Return[0]
G = np.eye(2)[np.argmax(G,axis=1)]
print(G - Y)
