#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
#%matplotlib inline
# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [1,10,14,34,44,36,22,67,79,90]
y = np.array(y)
#Show the effect of a scatter plot
plt.scatter(x,y)
plt.show()

#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b
def model(a, b, x):
  return a*x + b
#The most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
  num = len(x)
  prediction=model(a,b,x)
  return (0.5/num) * (np.square(prediction-y)).sum()
#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
  num = len(x)
  prediction = model(a,b,x)
  #Update the values of A and B by finding the partial derivatives of the loss function on a and b
  da = (1.0/num) * ((prediction -y)*x).sum()
  db = (1.0/num) * ((prediction -y).sum())
  a = a - Lr*da
  b = b - Lr*db
  return a, b
#iterated function, return a and b
def iterate(a,b,x,y,times):
  for i in range(times):
    a,b = optimize(a,b,x,y)
  return a,b

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 1e-4
#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,10)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
plt.show()

a,b = iterate(a,b,x,y,100)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
plt.show()

a,b = iterate(a,b,x,y,100000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
plt.show()