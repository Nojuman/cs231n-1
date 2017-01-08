import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  # TODO better way to do indexing?
  truth_idx = (tuple(xrange(num_train)),tuple(y))
  correct_scores = scores[truth_idx].reshape((num_train,1))
  margin = scores - correct_scores + 1
  margin[truth_idx] = 0

  loss += margin[margin > 0].sum() / num_train

  # positive term is easy - wherever the margin is positive, we want to add values from X.
  # in particular, for each class, sum X over the training cases where the margin is positive
  # for that class relative to the true class.
  positive = margin > 0
  # negative term is harder. remember that the negative contribution has the same total value
  # as the positive contribution, just shifted to different elements. in particular, all the mass
  # is shifted to the true class for each training case.
  negative = np.zeros((num_train, num_classes))
  negative[truth_idx] = positive.sum(axis=1)

  dW += X.T.dot(positive - negative) / num_train

  # add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
