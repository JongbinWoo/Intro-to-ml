import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input=np.empty(shape=(1, 1)), label=np.empty(shape=(1,1))):
        self.theta = np.random.randn(input.shape[1], label.shape[1])/np.sqrt(input.shape[1]/2) #Xaiver initialization
        self.input = input
        self.label = label
        self.lamda = 0

    def loss(self, input, label):
        return np.sum((np.matmul(input, self.theta)-label)**2)/2

    def predict(self, input):
        return np.matmul(input, self.theta)

    def bgd(self, input, label, learning_rate, training_epoch, lamda=0):
        self.lamda=lamda
        self.learning_rate=learning_rate
        for i in range(training_epoch):
            for n in range(input.shape[1]):
                sum=0
                for k in range(input.shape[0]):
                    sum += (np.matmul(input[k,:], self.theta)-label[k, :])*input[k, n]
                for l in range(label.shape[1]):
                    self.theta[n, l] -= learning_rate*(sum-lamda*(self.theta[n, l]))

    def sgd(self, input, label, learning_rate, training_epoch, lamda=0):
        self.lamda=lamda
        self.learning_rate = learning_rate
        for i in range(training_epoch):
            for n in range(input.shape[1]):
                sum = 0
                a=random.randint(0, input.shape[0]-1)
                sum += (np.matmul(input[a,:], self.theta)-label[a, :])*input[a, n]
                for l in range(label.shape[1]):
                    self.theta[n, l] -= learning_rate*(sum-lamda*(self.theta[n,l]))

    def closed(self, input, label, lamda=0):
        self.lamda=lamda
        mat=np.identity(n=input.shape[1])
        mat[0, 0] = 0
        self.theta=np.matmul(np.matmul(np.linalg.inv(np.dot(input.transpose(), input)+lamda*mat), input.transpose()), label)


def sort(train_data):
    for i in range(len(train_data)):
        for k in range(len(train_data)):
            if k!=0 and train_data[k, 1]<train_data[k-1, 1]:
                temp=np.copy(train_data[k,:])
                train_data[k, :]=train_data[k-1, :]
                train_data[k-1,:]=temp

np.random.seed(777)
#데이터 읽어오고 Test와 Train Set으로 나누기 전에 형태를 맞춰줌
data_name = "./data_lab2_iis.txt"
columns = ['x', 'y']
data = pd.read_csv(data_name, names=columns, sep=' ')
x_1 = np.asarray(data['x'])
x = np.expand_dims(x_1, axis=1)
a = np.ones_like(x)
x = np.concatenate([a, x], axis=1)
y = np.asarray(data['y'])
y = np.expand_dims(y, axis=1)

#feature를 5개까지 쓰므로 늘려줌

input_data=np.concatenate([x, np.expand_dims(x[:, 1]**2, axis=1)], axis= -1)
input_data=np.concatenate([input_data, np.expand_dims(x[:, 1]**3, axis=1)], axis= -1)
input_data=np.concatenate([input_data, np.expand_dims(x[:, 1]**4, axis=1)], axis= -1)
input_data=np.concatenate([input_data, np.expand_dims(x[:, 1]**5, axis=1)], axis= -1)
input_data=np.concatenate([input_data, y], axis=-1)
cut = int(len(input_data)*0.7)
#np.random.shuffle(input_data)
train_data=input_data[:cut,:]
test_data=input_data[cut:,:]
sort(train_data)
sort(test_data)

#1
x_train1=train_data[:,:2]
y_train1=train_data[:, -1]
x_test1=test_data[:,:2]
y_test1=test_data[:,-1]
y_train1=np.expand_dims(y_train1, axis=1)
y_test1=np.expand_dims(y_test1, axis=1)
#2
x_train2=train_data[:,:3]
y_train2=train_data[:, -1]
x_test2=test_data[:,:3]
y_test2=test_data[:,-1]
y_train2=np.expand_dims(y_train2, axis=1)
y_test2=np.expand_dims(y_test2, axis=1)
#3
x_train3=train_data[:,:6]
y_train3=train_data[:, -1]
x_test3=test_data[:,:6]
y_test3=test_data[:,-1]
y_train3=np.expand_dims(y_train3, axis=1)
y_test3=np.expand_dims(y_test3, axis=1)

#RIDGE를 위한 lamda정의
lamda=0.01

nonreg=Layer(x_train1, y_train1)
nonreg.closed(nonreg.input, nonreg.label)
parabolic=Layer(x_train2, y_train2)
parabolic.closed(parabolic.input, parabolic.label)
fiveth=Layer(x_train3, y_train3)
fiveth.closed(fiveth.input, fiveth.label)
#fiveth.bgd(fiveth.input, fiveth.label, 0.0001, 100, 0.1)
RIDGE=Layer(x_train3, y_train3)
RIDGE.closed(RIDGE.input, RIDGE.label, 0.1)
#RIDGE.bgd(RIDGE.input, RIDGE.label, 0.0001, 100, 0.1)

plt.plot(train_data[:,1],train_data[:,-1], "yo", label="Train Data")
plt.plot(x_train1[:,1], nonreg.predict(x_train1), label="Nonregularized")
plt.plot(x_train2[:,1], parabolic.predict((x_train2)), label="Parabolic")
plt.plot(x_train3[:,1], fiveth.predict(x_train3), label="5th")
plt.plot(x_train3[:,1], RIDGE.predict(x_train3), label="RIDGE")

#plt.plot(x_test, y_closedform, label="Test Predict")
plt.legend()
plt.show()

print("Nonregularized linear error and Optimal theta : {}\n {}".format(nonreg.loss(nonreg.input, nonreg.label), nonreg.theta))
print("Nonregularized parabolic error and Optimal theta : {}\n {}".format(parabolic.loss(parabolic.input, parabolic.label), parabolic.theta))
print("Nonregularized 5th error and Optimal theta : {}\n {}".format(fiveth.loss(fiveth.input, fiveth.label), fiveth.theta))
print("Regularized error by RIDGE and Optimal theta: {}\n {} by lamda : {}".format(RIDGE.loss(RIDGE.input, RIDGE.label), RIDGE.theta, RIDGE.lamda))


#print("Closed form loss : {}".format(loss(x_train2, y_train2, theta_nonregularized_closed)))
plt.plot(test_data[:,1],test_data[:,-1], "bo", label="Test Data")
plt.plot(x_test1[:,1], nonreg.predict(x_test1), label="Nonregularized Test")

plt.plot(x_test2[:,1], parabolic.predict(x_test2), label="Parabolic Test")
plt.plot(x_test3[:,1], fiveth.predict(x_test3), label="5th Test")
plt.plot(x_test3[:,1], RIDGE.predict(x_test3), label="RIDGE Test")
plt.legend()
plt.show()

print("Nonregularized linear Test error and Optimal theta : {}\n {}".format(nonreg.loss(x_test1, y_test1), nonreg.theta))
print("Nonregularized parabolic Test error and Optimal theta : {}\n {}".format(parabolic.loss(x_test2, y_test2), parabolic.theta))
print("Nonregularized 5th Test error and Optimal theta : {}\n {}".format(fiveth.loss(x_test3, y_test3), fiveth.theta))
print("Regularized Test error by RIDGE and Optimal theta: {}\n {} by lamda : {}".format(RIDGE.loss(x_test3, y_test3), RIDGE.theta, RIDGE.lamda))

plt.plot(train_data[:, 1], train_data[:, 6], "ro")
plt.plot(test_data[:, 1], test_data[:, 6], "bo")
plt.show()
