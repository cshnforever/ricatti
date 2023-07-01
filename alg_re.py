import numpy as np

def hamilton(A,B,Q,R):
    m=A.shape[0]
    
    x12=-np.matmul(np.matmul(B,np.linalg.inv(R)),np.transpose(B))

    ham=np.zeros((2*m,2*m))
    ham[0:m,0:m]=A
    ham[0:m,m:]=x12
    ham[m:,0:m]=-Q
    ham[m:,m:]=-np.transpose(A)

    return ham

def are(A,B,Q,R):
    m=A.shape[0]
    h=hamilton(A,B,Q,R)
    eigval,eigvec=np.linalg.eig(h)
    #print(eigval)
    #print(eigvec)
    z=np.zeros((2*m,m),dtype=np.complex_)
    k=0
    for i in range(len(eigval)):
        if eigval[i].real<0:
            z[:,k]=eigvec[:,i]
            k+=1
    x=z[:m,:]
    y=z[m:,:]
    p=np.matmul(y,np.linalg.inv(x))
    return p

def lqr_are(A,B,Q,R):
    P=are(A,B,Q,R)
    k=np.matmul(np.matmul(np.linalg.inv(R),np.transpose(B)),P).real
    return k