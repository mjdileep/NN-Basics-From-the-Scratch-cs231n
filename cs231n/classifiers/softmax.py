from builtins import range
from ssl import SSL_ERROR_EOF
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #print("X shape:", X.shape)
    #print("W shape:", W.shape)
    for i in range(X.shape[0]):
      x = X[i,:]
      scores = []
      for j in range(W.shape[1]):
        score = 0.0
        for k in range(W.shape[0]):
          score += W[k, j]*x[k]
        scores.append(np.exp(score))

      scores = np.array(scores)/np.sum(scores)
      loss -= np.log(scores[y[i]])
      
      dW[:,y[i]] -= x
      dW += x.reshape(W.shape[0],1)*scores

    dW /= X.shape[0]
    loss /= X.shape[0]

    loss += np.sum(W**2)*reg
    dW += np.sum(W)*2*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.exp(X.dot(W))
    scores_normalized = scores/scores.sum(axis=1).reshape(X.shape[0], 1)
    loss = -np.sum(np.log(scores_normalized[np.arange(X.shape[0]),y]))/X.shape[0]
    scores_normalized[np.arange(X.shape[0]),y]-=1
    dW = X.T.dot(scores_normalized)/X.shape[0]

    loss += np.sum(W**2)*reg
    dW += np.sum(W)*2*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
