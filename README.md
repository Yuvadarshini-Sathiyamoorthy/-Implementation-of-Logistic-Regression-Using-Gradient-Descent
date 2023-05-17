# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.
## Program:

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Yuvadarshini S
RegisterNumber: 212221230126 
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(x[y == 1][:, 0],x[y == 1][:, 1], label="Admitted",color="#fc0345")
plt.scatter(x[y == 0][:, 0],x[y == 0][:, 1], label="Not Admitted",color="purple")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()


plt.show()

    return 1 / (1 + np.exp(-z))
    
plt.plot()
x_plot = np.linspace(-10, 10, 100)
plt.plot(x_plot, sigmoid(x_plot),color="#c21d8b")
plt.show

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    x_plot=np.c_[xx.ravel(),yy.ravel()]
    x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
    y_plot=np.dot(x_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="#3F00FE")
    plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="#BFFE00")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
```

## Output:

## Array value of x

![51](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/396c4d7c-3fe9-4d99-be2b-7891bb80890c)

## Array Value of y

![52](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/ba7c5f82-0dc1-46c9-9984-dda5f4cc6651)

## Exam 1- Score graph

![53](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/7be16f61-2e3a-4a7e-b79a-7805e6ec3959)

## Sigmoid Function Graph

![54](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/56336c34-481b-46e1-8c3f-930700a71e05)

## X_train_grad value

![55](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/5aae14b9-dc20-4bf1-b906-62020355c2bd)

## Y_train_grad value

![56](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/6362d598-9ce0-404a-9444-d9fac522f98e)

## Print res.x

![57](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/d1bffadc-5538-4bd4-ba1b-88b6d9d9cbea)

## Decision Boundary grapg for Exam Score

![58](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/a4ba8cb8-1713-474c-977b-05b92a7a1e52)

## Probability value

![59](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/8e086168-f882-446b-a164-f09337fd5883)

## Prediction value of mean

![510](https://github.com/Yuvadarshini-Sathiyamoorthy/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/93482485/02f86cd1-efbd-4fb4-acb3-a62ff3ad5f35)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

