import operator
from functools import reduce
def c(n,k):
    return reduce(operator.mul,range(n-k+1,n+1))/reduce(operator.mul,range(1,k+1))

#baggin弱分类器叠加
def baggin(n,p):
    s=0.
    for i in range(int(n/2+1),n+1):
        s+=c(n,i)* p**i * (1-p)**(n-i)
    return s
if __name__=="__main__":
    for t in range(9,101,10):
        print(str(t)+'次采样正确率: %f'%(baggin(t,0.6)))