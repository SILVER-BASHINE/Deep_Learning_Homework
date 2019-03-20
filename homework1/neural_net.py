from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
class ThreeLayerNet(object):
  """
  A three-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses ReLU nonlinearities after the first and the second fully
  connected layers.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the third fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)
    
  
  def get_param(self):
        
    return self.params


 

    

  def loss(self, X, y=None, reg=0.0):  
   
    """
    Compute the loss and gradients for a three layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape

    
    scores = None
    T=np.dot(X,W1)+b1
    T1=np.maximum(T,0)
    T=np.dot(T1,W2)+b2
    T2=np.maximum(T,0)
    scores=np.dot(T2,W3)+b3
    #############################################################################
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores
    # Compute the loss
    loss = None
 

   #生成全0矩阵
    exp_scores=np.exp(scores)
    exp_scores/=(np.sum(exp_scores,axis=1).reshape(N,1))
    loss=-(1/N)*(np.sum(np.log(exp_scores[np.arange(N),y])))+0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(W2*W2)+0.5*reg*np.sum(W3*W3)

    delta_S=np.zeros_like(exp_scores)
    delta_S[range(N),y]+=1
    delta_S-=exp_scores
    grads = {}
    
    grads['W3']=reg*W3+(-1/N)*np.dot(T2.T,delta_S)
    grads['b3']=-(1/N)*np.sum(delta_S,axis=0)
    
    
    delta_t2=np.zeros_like(T2)
    delta_t2[T2>0]=1
    
    
    grads['W2']=reg*W2+(-1/N)*np.dot(T1.T,np.dot(delta_S,W3.T)*delta_t2)
    grads['b2']=(-1/N)*np.sum(np.dot(delta_S,W3.T)*delta_t2,axis=0)
    
    delta_t1 = np.zeros_like(T1)    
    
    zhenghe=(np.dot(delta_S,W3.T))*delta_t2  
    delta_t1[T1>0]=1    
    
    grads['W1']=reg*W1+(-1/N)*np.dot(X.T,np.dot(zhenghe,W2.T)*delta_t1)    
    grads['b1']=(-1/N)*np.sum(np.dot(zhenghe,W2.T)*delta_t1,axis=0)  
    return loss,grads
    

 


  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        r=np.random.choice(num_train,batch_size)
        X_batch=X[r,:]
        y_batch=y[r]
        
      # Compute loss and gradients using the current minibatch
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)

        self.params['W1']-=learning_rate*grads['W1']
        self.params['W2']-=learning_rate*grads['W2']
        self.params['W3']-=learning_rate*grads['W3']
        self.params['b1']-=learning_rate*grads['b1']
        self.params['b2']-=learning_rate*grads['b2']
        self.params['b3']-=learning_rate*grads['b3']

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

         # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
        # Check accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

        # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
           }


  def predict(self, X):
        y_pred = None
        score=self.loss(X)
        y_pred=np.argmax(score,axis=1)               
        return y_pred
  


