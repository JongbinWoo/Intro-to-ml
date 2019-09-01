import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['x','y']
training_file ='./data_lab1_iis.txt'
data_in = pd.read_csv(training_file, names=columns, sep = ' ')

test_file ='./custom_data.txt'
data_test = pd.read_csv(test_file, names=columns, sep = ' ')


data_in.plot(kind='scatter', x='x', y='y', color='red')
#plt.figure(5)
#plt.plot(x,y,'ro')
#plt.xlabel('x')
#plt.ylabel('y')


x = np.asarray(data_in['x'])
x = np.expand_dims(x,axis=1)
a = np.ones_like(x)
x_input = np.concatenate((a,x), axis = 1)
y = np.asarray(data_in['y'])
y = np.expand_dims(y,axis=1)


x_test = np.asarray(data_test['x'])
x_test = np.expand_dims(x_test,axis=1)
a_test = np.ones_like(x_test)
x_input_test = np.concatenate((a_test,x_test), axis = 1)
y_test = np.asarray(data_test['y'])

np.random.seed(777)

theta1 = np.random.randn(2, 1)
theta2 = np.random.randn(2, 1)
theta3 = np.random.randn(2, 1)
learning_rate = 0.001

#Batch gradient descent
for j in range(0,1000):
    for n in range(0,2):
        x0 = np.expand_dims(x_input[:,n],axis=1)
        theta1[n,0] = theta1[n,0] - learning_rate * np.sum((np.matmul(x_input, theta1) - y) * x0)
        if(j%100==0):
            print(np.sum(np.matmul(x_input, theta1) - y) ** 2)

#Stochastic gradient descent
for j in range(0,1000):
    for n in range(0,2):
        i = np.random.randint(100)
        theta2[n,0] = theta2[n,0] - learning_rate * (np.matmul(x_input[i,:],theta2) - y[i,:]) * x_input[i,n]
        if(j%100==0):
            print(np.sum(np.matmul(x_input, theta2) - y) ** 2)

#Ols
theta3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_input),x_input)),np.transpose(x_input)),y)


print(theta1)
y_predict_1 = np.matmul(x_input, theta1)


print(theta2)
y_predict_2 = np.matmul(x_input, theta2)

print(theta3)
y_predict_3 = np.matmul(x_input, theta3)

plt.plot(x, y_predict_1, label="BGD") # plotting t, a separately
plt.plot(x, y_predict_2, label="SGD")
plt.plot(x, y_predict_3, label="OLS")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


y_predict_4 = np.matmul(x_input_test, theta3)

plt.plot(x_test, y_predict_4, label="TEST")
plt.plot(x_test,y_test,'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
