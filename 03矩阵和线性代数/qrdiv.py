import numpy as np
import math
def is_same(a,b):
    n=len(a)
    for i in range(n):
        if math.fabs(a[i]-b[i])>1e-6:
            return False
    return True
if __name__=="__main__":
    a=np.array([0.65,0.28,0.07,0.15,0.67,0.18,0.12,0.36,0.52])
    n=math.sqrt(len(a))
    a=a.reshape((n,n))
    value,v=np.linalg.eig( a )
    """value是特征值 v是特征向量"""
    #help(np.linalg.eig)
    print(value,v)
    times=0
    help(np.diag)
    while (times==0) or (not is_same(np.diag(a),v)):
        v=np.diag(a)#[0.65,0.67,0.52]
        q,r=np.linalg.qr(a)
        a=np.dot(r,q)
        times+=1
        print("正交阵： ",q)
        print("三角阵： ",r)
        print("近似阵： ",a)
    print("times: ",times)
    print("精确特征值：",value)
    print(np.linalg.eig(a))
