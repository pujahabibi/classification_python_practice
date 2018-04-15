import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)
dataset = pd.read_csv("pima-indians-diabetes.csv")
X = dataset[["a", "b", "c", "d", 'e', "f", "g", "h"]]
Y = dataset[["label"]]
X = X.values
Y = Y.values

# normalize the X
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# X = np.asarray(X)
# Y = np.asarray(Y)
#setup weight and bias which connected to each layer or neurons

#input layer is connected to 10 neurons of hidden layer
w1 = np.random.rand(8, 10) * 0.01
b1 = np.zeros((1, 10))
#10 neurons hidden layer are connected to ouput layer
w2 = np.random.rand(10, 1) * 0.01
b2 = np.zeros((1, 1))

#make sigmoid function
def sigmoid(X, w, b):
    z = np.dot(X, w) + b
    return 1/(1 + np.exp(-z))

#do forward propagation in neural network
def forwardProp(X, w1, w2, b1, b2):
    hiddenZ = sigmoid(X, w1, b1)
    outputZ = sigmoid(hiddenZ, w2, b2)
    return hiddenZ, outputZ

#to calculate the cost function
def costFunction(Y, outputResult):
    m = X.shape[0]
    J = -np.sum(Y * np.log(outputResult) + (1 - Y) * np.log(1 - outputResult))/m
    return J

def derivativeOfW2(hiddenLayer, outputLayer, targetResult):
    m = targetResult.shape[0]
    dzOutputLayer = outputLayer - targetResult
    return (np.dot(hiddenLayer.T, dzOutputLayer))/m

def derivativeOfB2(outputLayer, targetResult):
    m = targetResult.shape[0]
    dzOutputLayer = outputLayer - targetResult
    return (np.sum(dzOutputLayer, axis=1, keepdims=True))/m

def derivativeOfW1(outputLayer, targetResult, newWeight2, hiddenLayer, X):
    m = X.shape[0]
    dzOutputLayer = outputLayer - targetResult
    dzHiddenLayer = np.dot(dzOutputLayer, newWeight2.T) * hiddenLayer * (1 - hiddenLayer)
    return (np.dot(X.T, dzHiddenLayer))/m

def derivativeOfB1(outputLayer, targetResult, newWeight2, hiddenLayer):
    m = targetResult.shape[0]
    dzOutputLayer = outputLayer - targetResult
    dzHiddenLayer = np.dot(dzOutputLayer, newWeight2.T) * hiddenLayer * (1 - hiddenLayer)
    return (np.sum(dzHiddenLayer, axis=1, keepdims=True))/m

# def fittingModel(X, Y, w1, w2, b1, b2, learningRate, iterations):
#     hiddenLayer, outputLayer = forwardProp(X, w1, w2, b1, b2)
#     costs = []
#     for i in range(iterations):
#         w2 = w2 - (learningRate * derivativeOfW2(hiddenLayer, outputLayer, Y))
#         b2 = b2 - (learningRate * derivativeOfB2(outputLayer, Y))
#         w1 = w1 - (learningRate * derivativeOfW1(outputLayer, Y, w2, hiddenLayer, X))
#         b1 = b1 - (learningRate * derivativeOfB1(outputLayer, Y, w2, hiddenLayer))
#
#         cost_result = costFunction(Y, outputLayer)
#         costs.append(cost_result)
#     return w1, w2, b1, b2, np.array(costs)
costs = []
for i in range(3000):
    hiddenLayer, outputLayer = forwardProp(X, w1, w2, b1, b2)
    w2 = w2 - (0.1 * derivativeOfW2(hiddenLayer, outputLayer, Y))
    b2 = b2 - (0.1 * derivativeOfB2(outputLayer, Y))
    w1 = w1 - (0.1 * derivativeOfW1(outputLayer, Y, w2, hiddenLayer, X))
    b1 = b1 - (0.1 * derivativeOfB1(outputLayer, Y, w2, hiddenLayer))

    cost_result = costFunction(Y, outputLayer)
    costs.append(cost_result)

plt.plot(costs)
plt.show()
newHidden, newOutput = forwardProp(X, w1, w2, b1, b2)
pred_result = np.where(newOutput >= 0.5, 1, 0)
accuracy_score = np.sum(Y == pred_result) / len(Y)
print(accuracy_score)
print(pred_result)

# hiddenLayer, outputLayer = forwardProp(X, w1, w2, b1, b2)
# newW1, newW2, newB1, newB2, costResult = fittingModel(X, Y, w1, w2, b1, b2, 1, 99999)
# newHidden, newOutput = forwardProp(X, newW1, newW2, newB1, newB2)
# pred_result = np.where(newOutput >= 0.5, 1, 0)
# accuracyScore = np.sum(Y == pred_result) / len(Y)
# print(accuracyScore)