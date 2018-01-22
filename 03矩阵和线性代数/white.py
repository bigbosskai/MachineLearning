import numpy as np
import math
def whiteing(x):
    m=len(x)
    n=len(x[0])
    #计算x*x'
    xx=[[0.0]*n for tt in range(n)]
    for i in range(n):
        for j in range(i,n):
            s=0.0
            for k in range(m):
                s+=x[k][i]*x[k][j]
            xx[i][j]=s
            xx[j][i]=s
    #xx是个对称阵
    #计算xx的特征值和特征向量
    lamda,egs=np.linalg.eig(xx)
    lamda=[1/math.sqrt(d) for d in lamda]
    #计算白话矩阵U'*^(-0.5)*U
    #计算U'和对角阵的乘积
    t=[[0.0]*n for tt in range(n)]
    for i in range(n):
        for j in range(n):
            t[i][j]=lamda[j]*egs[i][j]
    whiten_matrix=[[0.1]*n for tt in range(n)]
    for i in range(n):
        for j in range(n):
            s=0.0
            for k in range(n):
                s+=t[i][k]*egs[j][k]
            whiten_matrix[i][j]=s
    #白化
    wx=[0.0]*n
    for i in range(m):
        for j in range(n):
            s=0.0
            for k in range(n):
                s+=x[i][k]*whiten_matrix[j][k]
            wx[j]=s
        x[i]=wx[:]
    return x


