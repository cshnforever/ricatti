import numpy as np

def next_step(A,x,t,h): #RungeKutta 4th
    x1=x
    t1=t
    k1=np.matmul(A,x1)
    x2=x1+h*k1/2
    t2=t+h/2
    k2=np.matmul(A,x2)
    x3=x1+h*k2/2
    t3=t+h/2
    k3=np.matmul(A,x3)
    x4=x1+h*k3
    t4=t+h
    k4=np.matmul(A,x4)

    next_x=x+h/6*(k1+2*k2+2*k3+k4)
    next_t=t+h

    return next_x,next_t


def left_mul(A):    #Expand Matrix
    m1=A.shape[0]
    m=m1//2

    x=np.zeros((2*m*m,2*m*m))

    for i in range(m1):
        for j in range(m1):
            x[i*m:(i+1)*m,j*m:(j+1)*m]=A[i][j]*np.eye(m)

    return x


def hamilton(A,B,Q,R):
    m=A.shape[0]
    
    x12=-np.matmul(np.matmul(B,np.linalg.inv(R)),np.transpose(B))

    ham=np.zeros((2*m,2*m))
    ham[0:m,0:m]=A
    ham[0:m,m:]=x12
    ham[m:,0:m]=-Q
    ham[m:,m:]=-np.transpose(A)

    x_mat=left_mul(ham)

    return x_mat

def diff_re(A,B,Q,R,final_t,h):
    m=A.shape[0]
    ext_ham=hamilton(A,B,Q,R)
    init_mat=np.concatenate((np.eye(m),Q),axis=0)
    init_mat=np.reshape(init_mat,(2*m*m,1))

    tt=np.linspace(0,final_t,(int)(final_t/h))
    p_ls=np.zeros((m,m,len(tt)))

    z=init_mat
    t=final_t
    for i in range(len(tt)):
        x=np.reshape(z[:m*m],(m,m))
        y=np.reshape(z[m*m:],(m,m))
        p_ls[:,:,len(tt)-1-i]=np.matmul(y,np.linalg.inv(x))
        z,t=next_step(ext_ham,z,t,-h)

    return p_ls


def lqr_diff(A,B,Q,R,t,h=0.1):
    m=A.shape[0]
    n=B.shape[1]
    p=diff_re(A,B,Q,R,t,h)
    l=len(p[0,0])
    k=np.zeros((n,m,l))
    temp=np.matmul(np.linalg.inv(R),np.transpose(B))
    for i in range(l):
        k[:,:,i]=np.matmul(temp,p[:,:,i])
    return k
