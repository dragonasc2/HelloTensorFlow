import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  logits = np.dot(X, W)
  h_1 = np.exp(logits)
  h_2 = h_1 / np.sum(h_1, axis=1, keepdims=True)
  dLogitsdW = X.T
  dLdLogits = np.zeros((X.shape[0], W.shape[1]))
  for idx in range(X.shape[0]):
    loss += - np.log(h_2[idx, y[idx]])
    dLdLogits[idx, :] = h_2[idx]
    dLdLogits[idx, y[idx]] -= 1

  loss /= X.shape[0]
  loss += np.sum(reg * W * W)
  dW = dLogitsdW.dot(dLdLogits) / X.shape[0] + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_one_hot = np.zeros((y.shape[0], W.shape[1]))
  idxs = np.arange(y.shape[0])
  y_one_hot[idxs, y] = 1

  logits = np.dot(X, W)
  h_1 = np.exp(logits)
  h_2 = h_1 / np.sum(h_1, axis=1, keepdims=True)
  loss = - np.sum(y_one_hot * np.log(h_2) / y.shape[0]) + np.sum(reg * W * W)
  dLdLogits = y_one_hot * (h_2 - 1) + (1 - y_one_hot) * (h_2)
  dLogitsdW = X.T
  dW = dLogitsdW.dot(dLdLogits) / y.shape[0] + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

