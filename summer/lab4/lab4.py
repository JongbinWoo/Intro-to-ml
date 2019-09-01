import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(777)
def data_generation():
    data = []
    for i in range(30):
        x = np.random.randint(2, size=8)
        y = np.sum(x)
        data_i = np.append(x,y)
        data += [data_i]
    return data

data = np.array(data_generation())

F = np.zeros_like(data, dtype = np.float32)
#np.random.seed(777)
#V1 = np.random.randn(2)
#np.random.seed(777)
#V2 = np.random.randn(2)
#np.random.seed(777)
#V3 = np.random.randn(2)
V1 = np.array([0.,0.])
V2 = np.array([0.,0.])
V3 = np.array([0.,0.])
def forward_propagation(V,data,F):
    Vf = V[0]
    Vx = V[1]
    for i in range(30):
        for t in range(8):
            F[i,t+1] = (F[i,t] * Vf) + (data[i,t] * Vx)
    return F

def derivate(V,data,F):
    dVf = 0
    dVx = 0
    for t in range(8):
        #for i in range(30):
        a = np.sum((forward_propagation(V,data,F)[:,8] -  data[:,8]) * F[:,t])
        b = np.sum((forward_propagation(V,data,F)[:,8] -  data[:,8]) * data[:,t])
        dVf += a * (V[0] ** (7 - t))
        dVx += b * (V[0] ** (7 - t))
    return dVf,dVx

def back_propagation(V,data,F,alpha):
    D = derivate(V,data,F)
    dVf = D[0]
    dVx = D[1]
    V[0] -= alpha * dVf
    V[1] -= alpha * dVx
    return V

def resilient(V,data,F, delta_x, delta_f):
    e_p = 1.2
    e_n =0.5
    #previous
    D = derivate(V,data,F)
    dVf = D[0]
    dVx = D[1]
    V[0] -= delta_f * np.sign(dVf)
    V[1] -= delta_x * np.sign(dVx)

    #current
    D = derivate(V,data,F)
    dVf_c = D[0]
    dVx_c = D[1]
    #update delta
    if(np.sign(dVf) * np.sign(dVf_c) ==1 ):
        delta_f = delta_f * e_p
    else:
        delta_f = delta_f * e_n
    if(np.sign(dVx) * np.sign(dVx_c) ==1 ):
        delta_x = delta_x * e_p
    else:
        delta_x = delta_x * e_n

    V[0] -= delta_f * np.sign(dVf)
    V[1] -= delta_x * np.sign(dVx)
    return V, delta_f, delta_x

def gradient_clipping(V,data,F):
    alpha = 0.0001
    e = 0.7
    D = derivate(V,data,F)
    dVf = D[0]
    dVx = D[1]
    if(dVf > e):
        dVf = e
    if(dVx > e):
        dVx = e
    V[0] -= alpha * dVf
    V[1] -= alpha * dVx
    return V

def error(V,data,F):
    E = np.sum((forward_propagation(V,data,F)[:,8] - data[:,8]) ** 2 )
    return E
########################################################
#Back prop
i=0
while(error(V1,data,F)>0.1):
    back_propagation(V1,data,F,0.0001)
    i+=1
#resilient prop
delta_x = 0.001
delta_f = 0.001
j=0
while(error(V2,data,F)>0.1):
    R = resilient(V2,data,F,delta_x,delta_f)
    delta_f = R[1]
    delta_x = R[2]
    j+=2

k=0
while(error(V3,data,F)>0.1):
    gradient_clipping(V3,data,F)
    k+=1

data[:,8]
V1
forward_propagation(V1,data,F)[:,8]
V2
forward_propagation(V2,data,F)[:,8]
V3
forward_propagation(V3,data,F)[:,8]

i
j
k

#################################################################
V1 = np.array([1.5,1.5])
V2 = np.array([1.5,1.5])
V3 = np.array([1.5,1.5])

#Back prop
for i in range(500):
    back_propagation(V1,data,F,0.0001)

#resilient prop
delta_x = 0.001
delta_f = 0.001
for i in range(100):
    R = resilient(V2,data,F,delta_x,delta_f)
    delta_f = R[1]
    delta_x = R[2]

for i in range(10000):
    gradient_clipping(V3,data,F)

V1
forward_propagation(V1,data,F)[:,8]
V2
forward_propagation(V2,data,F)[:,8]
V3
forward_propagation(V3,data,F)[:,8]
##################################################################
np.random.seed(666)
test = np.array(data_generation())

print(test[:,8])
forward_propagation(V1,test,F)[:,8]
forward_propagation(V2,test,F)[:,8]
forward_propagation(V3,test,F)[:,8]
