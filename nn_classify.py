#!/usr/bin/env python

from __future__ import division

import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def sigmoid_gradient(x):
  g = sigmoid(x)
  return np.multiply(g, 1 - g)

class NN_classify:
  """ This takes in an input matrix and label matrix and tries to 
      predict the label y given an input X. This is a classifier."""

  def __init__(self, nodes_per_layer):
    """ 
    @param nodes_per_layer: The length of the list corresponds to the number
                            of layers of the neural network. For example,
                            [2, 3, 4] implies an input layer of 2 nodes,
                            a hidden layer of 3 nodes and an output layer
                            of 4 nodes.
    """
    self.__theta_lst__ = []
    for i in range(1, len(nodes_per_layer)):
      prev, curr = nodes_per_layer[i-1], nodes_per_layer[i]
      self.__theta_lst__.append(np.matrix(np.random.rand(curr, prev), dtype=np.complex64))

  def train(self, X, y, alpha=0.3, num_iter=100, verbose=False):
    """
    @param X:        This is a matrix whose rows correspond to samples.
    @param y:        This is the label matrix whose row represents the label
                     of the corresponding row in X.
    @param alpha:    This controls the rate of the gradient descent.
    @param num_iter: Number of iterations.
    @param verbose:  Set this to true for printed updates.
    """
    np.seterr(all='raise')
    c = [self.__cost__(X, y)]
    if verbose:
      print "Init cost: %f" % c[-1]
    for i in range(num_iter):
      grad = self.__theta_gradients__(X, y)
      k = len(self.__theta_lst__)
      for j in range(k):
        self.__theta_lst__[j] -= alpha * grad[j]
      c.append(self.__cost__(X, y))
      if verbose:
        print "Progress: %d / %d, Cost: %f" % (i+1, num_iter, c[-1])
    return np.array(c, dtype=np.complex64)

  def predict(self, X, threshold=0.5):
    """
    @param X:           Input.
    @param threshold:   Cutting point to decide between classes.
    """
    return (self.__get_activation_layers__(X)[-1] > threshold).astype(int)

  def __get_activation_layers__(self, X):
    activation_layers = [X]
    for theta in self.__theta_lst__:
      activation_layers.append(sigmoid(activation_layers[-1] * theta.T))
    return activation_layers

  def __cost__(self, X, y):
    h = self.__get_activation_layers__(X)[-1]
    return np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1-h)))

  def __theta_gradients__(self, X, y):
    activation_layers = [X]; z_lst = []
    idx = 0
    m, n = X.shape
    for theta in self.__theta_lst__:
      z_lst.append((idx, activation_layers[-1] * theta.T))
      activation_layers.append(sigmoid(z_lst[-1][1]))
      idx += 1
    delta = [activation_layers[-1] - y]
    z_lst.pop()
    for idx, z in z_lst[::-1]:
      delta.insert(0, np.multiply(delta[0] * self.__theta_lst__[idx + 1], sigmoid_gradient(z)))
    idx = 0; grad = []
    for a in activation_layers[:-1:]: 
      grad.append((delta[idx].T * a)/m)
      idx += 1
    return grad

if __name__ == '__main__':

  nn = NN_classify([2, 3]) #[1, 0, 0] if left > right, [0, 1, 0] if equal, [0, 0, 1] if right > left

  X = np.matrix('[0 1; 1 0; 1 1; 5 1; 2 6; 3 5; 2 2; 2 4]')
  y = np.matrix('[0 0 1; 1 0 0; 0 1 0; 1 0 0; 0 0 1; 0 0 1; 0 1 0; 0 0 1]')

  c = nn.train(X, y, alpha=0.3, num_iter=500, verbose=False)

  print "Plotting cost."
  print "Close figure to continue."

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(abs(c))
  plt.xlabel('Iterations')
  plt.ylabel('Cost')
  plt.show()

  print "Threshold: %f" % 0.5

  print "Expecting:"
  print "[[0 0 1]"
  print " [1 0 0]]"
  print "Got:"
  print nn.predict(np.matrix('[5 10; 15 6]'), 0.5)
