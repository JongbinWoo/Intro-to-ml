import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x,theta):
    return np.matmul(x,theta)

def error(x,y,theta):
    return np.sum((np.matmul(x,theta)-y) ** 2)

def makeH1(a,x):
    return np.concatenate((a,x), axis =1)

def makeH2(a,x):
    return np.concatenate((makeH1(a,x),x**2),axis=1)

def makeH3(a,x):
    return np.concatenate((makeH2(a,x),x**3,x**4,x**5),axis=1)

columns = ['x','y']
name_file ='./data_lab2_iis.txt'
data_in = pd.read_csv(name_file, names=columns, sep = ' ')

split = int(len(data_in) * 0.7)
print(split)

x = np.asarray(data_in['x'])
x = np.expand_dims(x,axis=1)

y = np.asarray(data_in['y'])
y = np.expand_dims(y,axis=1)

#data shuffling
data = np.concatenate((x,y), axis=1)
np.random.shuffle(data)
#print(data)
#split training, test data
#rint(np.expand_dims(data[:split, 0],axis=1))
training_data_x = np.sort(np.expand_dims(data[:split, 0],axis=1),axis=0)
training_data_y = np.sort(np.expand_dims(data[:split, 1],axis=1),axis=0)

test_data_x = np.sort(np.expand_dims(data[split+1:,0],axis=1),axis=0)
test_data_y = np.sort(np.expand_dims(data[split+1:,1],axis=1),axis=0)
#print(training_data_x)
a = np.ones_like(training_data_x)
a_t = np.ones_like(test_data_x)

x_h1_training = makeH1(a,training_data_x)
theta1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_h1_training),x_h1_training)),np.transpose(x_h1_training)),training_data_y)

x_h2_training = makeH2(a,training_data_x)

theta2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_h2_training),x_h2_training)),np.transpose(x_h2_training)),training_data_y)


x_h3_training = makeH3(a,training_data_x)
theta3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_h3_training),x_h3_training)),np.transpose(x_h3_training)),training_data_y)

lamda = 0.000001
x_h4_training = makeH3(a,training_data_x)
m = np.identity(n=len(x_h4_training[0]))
m[0,0] = 0
theta4 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_h4_training),x_h4_training)) + lamda * m,np.transpose(x_h4_training)),training_data_y)

predict_h1 = predict(x_h1_training,theta1)
predict_h2 = predict(x_h2_training,theta2)
predict_h3 = predict(x_h3_training,theta3)
predict_h4 = predict(x_h4_training,theta4)

plt.plot(training_data_x,training_data_y,'ro')
plt.plot(training_data_x, predict_h1, label="unregularized linear") # plotting t, a separately
plt.plot(training_data_x, predict_h2, label="unregularized parabolic")
plt.plot(training_data_x, predict_h3, label="unregularized 5th-order polynomial")
plt.plot(training_data_x, predict_h4, label="regularized 5th-order polynomial")
plt.ylim(0, 20)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

training_error_h1 = error(x_h1_training,training_data_y,theta1)
training_error_h2 = error(x_h2_training,training_data_y,theta2)
training_error_h3 = error(x_h3_training,training_data_y,theta3)
training_error_h4 = error(x_h4_training,training_data_y,theta4)
print(training_error_h1,training_error_h2,training_error_h3,training_error_h4)

plt.plot(test_data_x,test_data_y,'ro')
plt.plot(test_data_x, predict(makeH1(a_t,test_data_x),theta1), label="unregularized linear") # plotting t, a separately
plt.plot(test_data_x, predict(makeH2(a_t,test_data_x),theta2), label="unregularized parabolic")
plt.plot(test_data_x, predict(makeH3(a_t,test_data_x),theta3), label="unregularized 5th-order polynomial")
plt.plot(test_data_x, predict(makeH3(a_t,test_data_x),theta4), label="regularized 5th-order polynomial")
plt.ylim(0, 20)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

test_error_h1 = error(makeH1(a_t,test_data_x),test_data_y,theta1)
test_error_h2 = error(makeH2(a_t,test_data_x),test_data_y,theta2)
test_error_h3 = error(makeH3(a_t,test_data_x),test_data_y,theta3)
test_error_h4 = error(makeH3(a_t,test_data_x),test_data_y,theta4)
print(test_error_h1,test_error_h2,test_error_h3,test_error_h4)
