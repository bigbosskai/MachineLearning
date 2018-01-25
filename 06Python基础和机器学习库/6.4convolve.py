import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
if __name__=="__main__":
    stock_max,stock_min,stock_close,stock_amount=np.loadtxt('6.SH600000.txt',delimiter='\t',skiprows=2,usecols=(2,3,4,5),unpack=True)
    N=100
    stock_close = stock_close[:N]
    print(stock_close)

    n=5
    weight=np.ones(5)
    weight/=weight.sum()
    print(weight)
    simple_stock_ema = np.convolve(stock_close,weight,mode="valid")# simple moving average

    weight = np.linspace(1,0,n)
    weight = np.exp(weight)
    weight/=weight.sum()
    print(weight)
    exponential_stock_ema = np.convolve(stock_close,weight,mode="vlid")# exponential moving average
    # print(len(stock_close))
    # print(len(stock_ema))
    t=np.arange(n-1,N)
    simple_poly= np.polyfit(t,simple_stock_ema,10)
    exponential_poly=np.polyfit(t,exponential_stock_ema,10)

    print(simple_poly)
    print(exponential_poly)

    simple_stock_ema_hat = np.polyval(simple_poly,t)
    exponential_stock_ema_hat=np.polyval(exponential_poly,t)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot(np.arange(N),stock_close,'ro-',linewidth=2,label=u'原始收盘价')
    t=np.arange(n-1,N)
    plt.plot(t,simple_stock_ema_hat,'b-',linewidth=2,label=u'简单移动平均线')
    plt.plot(t,exponential_stock_ema_hat,'g-',linewidth=2,label=u"指数滑动平均线")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig("convolve.png")
    plt.show()

    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
    # plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    # plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
    # plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.show()
