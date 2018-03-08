import numpy as np

def standard_scaler(data):
    """
    :param data:data is [n_samples,n_features]
    :return: standard_scaler data the mean is zero std is 1
    """
    """
        from sklearn import preprocessing
        import numpy as np
        # shape of data :[n_samples,n_features]
        data=np.array([[2,-1,2],[1,3,-1]])
        scaler=preprocessing.StandardScaler()
        scaler.fit(data)
        trans_data=scaler.transform(data)
        print('original data')
        print(data)
        print('transformed data')
        print(trans_data)
        print('use numpy by myself')
        mean=np.mean(data,axis=0)
        print(mean)
        std=np.std(data,axis=0)
        print('std')
        print(std)
        print(np.std([2,1]))
        print((((2.0-1.5)**2+(1.0-1.5)**2)/2.0)**0.5)
        var=std*std
        another_trans_data=data-mean
        print(another_trans_data)
        another_trans_data=another_trans_data/std
        print(another_trans_data)
        """
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    a_data=data-mean
    a_data=a_data/std
    return a_data