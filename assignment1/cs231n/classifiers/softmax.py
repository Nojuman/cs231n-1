import numpy as np
from random import shuffle

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

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # normalize to avoid blowup
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    sum_score_exp = 0.0
    for j in xrange(num_classes):
        sum_score_exp += np.exp(scores[j])
    loss += np.log(sum_score_exp) - correct_class_score

    for j in xrange(num_classes):
        dW[:,j] += X[i,:] * np.exp(scores[j]) / sum_score_exp
        if j == y[i]:
            dW[:,j] -= X[i,:]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  # normalize to avoid blowup
  scores -= np.max(scores)

  truth_idx = (tuple(xrange(num_train)),tuple(y))
  correct_scores = scores[truth_idx].reshape((num_train,1))

  score_exp = np.exp(scores)
  sum_score_exp = score_exp.sum(axis=1)
  loss += np.sum(np.log(sum_score_exp) - correct_scores) / num_train

  positive = score_exp / sum_score_exp.reshape((num_train,1))
  negative = np.zeros((num_train, num_classes))
  negative[truth_idx] = 1

  dW += X.T.dot(positive - negative)

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
