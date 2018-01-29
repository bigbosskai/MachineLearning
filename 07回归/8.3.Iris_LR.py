import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]
if __name__=="__main__":
    path = u'8.iris.data'  # 数据文件路径
    #
    # data = np.loadtxt(path, dtype=float, delimiter=',')
    # print(data)
    df=pd.read_csv(path,header=0)
    print(type(df))
    # print(df) DataFrame
    x=df.values[:,:-1]
    y=df.values[:,-1]
    x = x[:, 2:4]
    # print(type(x)) ndarray
    # print(type(y)) ndarray
    le=preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    print(le.classes_)
    y=le.transform(y)
    # print(y)
    #
    # x = StandardScaler().fit_transform(x)
    # lr = LogisticRegression()   # Logistic回归模型
    #     lr.fit(x, y.ravel())        # 根据数据[x,y]，计算回归参数
    #
    # 等价形式
    lr=Pipeline(
        [('sc',StandardScaler()),
         ('clf',LogisticRegression())]
    )
    # print(type(y))
    # print(type(y.ravel()))
    # print(y.shape)
    # print(y.ravel().shape)
    lr.fit(x,y.ravel())

    #paint
    N,M=500,500
    x1_min,x1_max=x[:,0].min(),x[:,0].max()
    x2_min,x2_max=x[:,1].min(),x[:,1].max()

    t1=np.linspace(x1_min,x1_max,N)
    t2=np.linspace(x2_min,x2_max,M)

    x1,x2=np.meshgrid(t1,t2)#生成一个网格
    x_test=np.stack((x1.flat,x2.flat),axis=1)
    print(type(x_test))

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    y_hat = lr.predict(x_test)                  # 预测值
    y_hat = y_hat.reshape(x1.shape)
    # 使之与输入的形状相同
    plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
    plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',s=50,cmap=cm_dark)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()
    #训练集上的预测结果
    y_hat=lr.predict(x)
    y=y.reshape(-1)
    result=y_hat==y
    print(y_hat)
    print(result)
    acc=np.mean(result)
    print(acc)
