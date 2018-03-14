#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    if s==b'Iris-setosa':
        return 0
    elif s==b'Iris-versicolor':
        return 1
    else:
        return 2


# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
# iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    print('load data success')
    x,y=np.split(data,[4,],axis=1)
    x=x[:,:2]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
    model=Pipeline([
        ('ss',StandardScaler()),
        ('DTC',DecisionTreeClassifier(criterion='entropy',max_depth=3))# criterion='gini'
    ])
    model.fit(x_train,y_train)
    y_test_hat=model.predict(x_test)
    f=open('iris_tree2018-3-8.dot','w')
    tree.export_graphviz(model.get_params('DTC')['DTC'],out_file=f)
    N,M=100,100
    x1_min,x1_max=x[:, 0].min(),x[:, 0].max()
    x2_min,x2_max=x[:, 1].min(),x[:, 1].max()
    t1=np.linspace(x1_min,x1_max,M)
    t2=np.linspace(x2_min,x2_max,N)
    x1,x2=np.meshgrid(t1,t2)
    x_show=np.stack((x1.flat,x2.flat),axis=1)
    # print(x_show.shape)
    print(x1)
    print(x2)
    print(x_show)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    y_show_hat = model.predict(x_show)  # 预测值
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o' )
    plt.scatter(x[:, 0],x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)
    plt.xlabel(iris_feature[0])
    plt.ylabel(iris_feature[1])
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.show()


    # acc
    y_test=y_test.reshape((1,-1))
    result=(y_test==y_test_hat)
    acc=np.mean(result)
    print("准确度: %.2f"%(acc))



    # find the best depth
    n_depths=list(range(1,15))
    errlist=[]
    for i in n_depths:
        model=Pipeline([
            ('ss',StandardScaler()),
            ('TDC',DecisionTreeClassifier(criterion='entropy',max_depth=i))
        ])
        model.fit(x_train,y_train)
        x_test_hat=model.predict(x_test)
        result=(x_test_hat==y_test)
        acc=np.mean(result)
        err=1-acc
        errlist.append(err)
        print("err rate:%.2f"%(err))

    plt.figure(facecolor='w')
    plt.plot(n_depths,errlist,'ro-',lw=2)
    plt.xlabel("决策树的深度")
    plt.ylabel("错误率")
    plt.title("决策树的深度与过拟合")
    plt.grid(True)
    plt.show()


