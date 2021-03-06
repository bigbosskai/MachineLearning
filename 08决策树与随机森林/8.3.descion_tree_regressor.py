import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor


if __name__=="__main__":
    N=100
    x=np.random.rand(N)*6-3  #[-3,3)
    # uniform distribution[0,1)
    # help(np.random.rand)
    # print(len(x))
    x.sort()
    y=np.sin(x)+np.random.randn(N)*0.05
    # standard normal" distribution.
    # Two - by - four array of samples from N(3, 6.25):
    # 2.5 * np.random.randn(2, 4) + 3
    # help(np.random.randn)
    x=x.reshape(-1,1)
    print(x.shape)
    # plt.figure(figsize=(10,10))
    # plt.plot(x,y,'-r')
    # plt.show()
    help(DecisionTreeRegressor)
    reg=DecisionTreeRegressor(criterion='mse',max_depth=9)
    dt=reg.fit(x,y)
    x_test=np.linspace(-3,3,50).reshape(-1,1)
    y_hat=dt.predict(x_test)
    plt.plot(x,y,'r*',linewidth=2,label='Actual')
    plt.plot(x_test,y_hat,'g-',linewidth=2,label='Predict')
    # plt.legend
    plt.grid()
    plt.savefig('single_tree_regressor.png')
    plt.show()




    # 比较决策树的深度影响
    plt.figure()
    depth=(2,4,6,8,10)
    clr='rgbmy'
    reg=(
        DecisionTreeRegressor(criterion='mse',max_depth=depth[0]),
        DecisionTreeRegressor(criterion='mse',max_depth=depth[1]),
        DecisionTreeRegressor(criterion='mse',max_depth=depth[2]),
        DecisionTreeRegressor(criterion='mse',max_depth=depth[3]),
        DecisionTreeRegressor(criterion='mse',max_depth=depth[4])
    )
    plt.plot(x,y,'k^',linewidth=2,label='Actual')
    x_test=np.linspace(-3,3,50).reshape(-1,1)
    for i,dt in enumerate(reg):
        clf=dt.fit(x,y)
        y_hat=clf.predict(x_test)
        plt.plot(x_test,y_hat,color=clr[i],linewidth=2,label="depth=%d"%(depth[i]))
    plt.legend(loc="upper left")
    plt.grid()
    # plt.sh
    plt.savefig('depth_inflence_on.png')
    plt.show()






