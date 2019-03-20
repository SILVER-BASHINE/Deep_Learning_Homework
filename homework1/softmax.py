import numpy as np

def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    m=X.shape[0]
    classes = max(y) + 1 ##类别数为最大数加1
    yt = np.zeros(shape=(y.shape[0],classes))##生成全0矩阵
    yt[np.arange(0,y.shape[0]),y] = 1
    yt= yt.astype('int64')
    
    A=np.exp(np.dot(X,W))
    s=np.reshape(np.sum(A,axis=1),(np.array(A).shape[0],-1))
    A=A/s
    loss=-1/m*(np.sum(np.log(A)*yt))+0.5*reg*np.sum(W*W)
    dW=-1/m*np.dot(X.T,(yt-A))+reg*W
    return loss, dW

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    m=X.shape[0]
    classes = max(y) + 1 ##类别数为最大数加1
    yt = np.zeros(shape=(y.shape[0],classes))##生成全0矩阵
    yt[np.arange(0,y.shape[0]),y] = 1
    yt= yt.astype('int64')
    
    A=np.dot(X,W)
    for i in range(np.array(A).shape[0]):
        A[i]=np.exp(A[i])/sum(np.exp(A[i]))
    a=np.log(A)
    J=0.0
    for i in range(np.array(A).shape[0]):
        J+=-(sum(a[i]*yt[i]))
    J=J/m+0.5*reg*np.sum(W*W)
    dW=reg*W+(-1/m)*np.dot(X.T,(yt-A))
    loss=J
    return loss, dW




