import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("test.txt")

plt.plot(data[:,0], data[:,1])

plt.show()
