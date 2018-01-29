import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,Ridge

if __name__=="__main__":
    """pandas read"""
    path = "8.Advertising.csv"
    data=pd.read_csv(path)
    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']
    # print(x)
    # print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
    model=Ridge()
    alpha_cans=np.logspace(-3,2,10)
    lasso_model=GridSearchCV(model,param_grid={'alpha':alpha_cans},cv=5)
    lasso_model.fit(x_train,y_train)
    print("paramters:\n")
    print(lasso_model.best_params_)

    y_hat=lasso_model.predict(x_test)
    mse=np.average( (y_hat-np.array(y_test))**2)
    rmse=np.sqrt(mse)
    print("mse",mse," rmse",rmse)

    t=np.arange(len(y_test))
    plt.plot(t,y_test,"r-",linewidth=2,label="Test")
    plt.plot(t,y_hat,"g-",linewidth=2,label="Predict")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('figureRidge.png')
    plt.show()