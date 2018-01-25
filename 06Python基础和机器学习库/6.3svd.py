import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

def restore(sigma,u,v,K):#奇异值、做特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m,n))#返回的新的矩阵
    for k in range(K):#默认sigma返回值就是从大到小的排列
        uk = u[:, k].reshape(m,1)#取第k列
        vk = v[k].reshape(1,n)
        a += sigma[k] * np.dot(uk,vk)
    a[a<0]=0
    a[a>255]=255
    # a = a.clip(0,255)
    return np.rint(a).astype('uint8')

if __name__=="__main__":
    A = Image.open('lena.png')
    output_path = r'.\LenaPic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    K=50
    #分离出来rgb三个矩阵
    u_r, sigma_r, v_r = np.linalg.svd( a[:,:,0] )
    u_g, sigma_g, v_g = np.linalg.svd( a[:,:,1] )
    u_b, sigma_b, v_b = np.linalg.svd( a[:,:,2] )
    plt.figure(figsize=(10,10),facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    print( u_r.shape )
    print( u_r )
    # print(u_r.shape,v_r.shape)
    # print( sigma_r )
    help( np.rint )
    for k in range(1,K+1):
        # print(k)
        R = restore( sigma_r, u_r, v_r, k)
        G = restore( sigma_g, u_g, v_g, k)
        B = restore( sigma_b, u_b, v_b, k)
        I = np.stack((R,G,B),axis=2)
        Image.fromarray(I).save('%s\\svd_%d.png'%(output_path,k))
        if k<=12:
            plt.subplot(3,4,k)
            plt.imshow(I)#传入的是RGB矩阵
            plt.axis('off')
            plt.title(u'奇异值个数：%d'%(k))
    plt.suptitle(u'SVD与图像分解',fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()