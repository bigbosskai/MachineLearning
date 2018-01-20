import matplotlib.pyplot as plt
import numpy
if __name__=="__main__":
    u=numpy.random.uniform(0.0,1.0,10000)
    plt.hist( u, bins=80, facecolor='green', alpha=0.5)
    plt.grid(True)
    plt.show()

    times=10000
    for time in range(times):
        u+=numpy.random.uniform(0.0,1.0,10000)
    print(len(u))
    u/=times
    plt.hist( u, bins=80, facecolor='red', alpha=0.75)
    plt.grid(True)
    plt.show()
