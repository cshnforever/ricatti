import numpy as np
import diff_re
import alg_re
import tfftn
import matplotlib.pyplot as plt

m=3
n=2

a=np.array([[1,-3,2],[4,3,0],[0,1,1]])
b=np.array([[-1,1],[-1,-1],[-1,2]])
q=np.diag([1,1,1])
r=np.diag([1,1])

t=5
h=0.01
tt=np.linspace(0,t,int(t/h)+1)
fig,ax = plt.subplots(3,2,figsize=(15,10))

###### Differential Ricatti Equation #########

k_ls=diff_re.lqr_diff(a,b,q,r,t,h)

x1_ls=np.zeros((m,len(tt)))
u1_ls=np.zeros((n,len(tt)))
x_0=np.array([[1],[1],[1]])
x=x_0
t_0=0
t1=t_0

for i in range(len(tt)-1):
    k=k_ls[:,:,i]
    s=a-np.matmul(b,k)
    u=-np.matmul(k,x)

    x1_ls[:,i]=np.squeeze(x)
    u1_ls[:,i]=np.squeeze(u)

    x,t1=diff_re.next_step(s,x,t1,h)

yy0=np.zeros((len(tt),1))
for i in range(3):
    ax[i][0].plot(tt,x1_ls[i,:],color='r')
    ax[i][0].plot(tt,yy0,alpha=0.5)
    ax[i][0].set_title(f'{i+1}th-variable')
for i in range(2):
    ax[i][1].plot(tt,u1_ls[i,:],color='r')
    ax[i][1].set_title(f'{i+1}th-input')

###### Algebraic Ricatti Equation #########

k=alg_re.lqr_are(a,b,q,r)
print(k)

x2_ls=np.zeros((m,len(tt)))
u2_ls=np.zeros((n,len(tt)))
x_0=np.array([[1],[1],[1]])
x=x_0
t_0=0
t1=t_0

for i in range(len(tt)-1):
    s=a-np.matmul(b,k)
    u=-np.matmul(k,x)

    x2_ls[:,i]=np.squeeze(x)
    u2_ls[:,i]=np.squeeze(u)

    x,t1=diff_re.next_step(s,x,t1,h)



yy0=np.zeros((len(tt),1))
for i in range(3):
    ax[i][0].plot(tt,x2_ls[i,:],color='b')
    #ax[i][0].plot(tt,yy0,alpha=0.3)
    ax[i][0].set_title(f'{i+1}th-variable')
for i in range(2):
    ax[i][1].plot(tt,u2_ls[i,:],color='b')
    ax[i][1].set_title(f'{i+1}th-input')

#fig.savefig("ricatti.jpg")


plt.show()