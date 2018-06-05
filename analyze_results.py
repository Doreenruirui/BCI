import matplotlib.pyplot as plt
import numpy as np


x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability')

x = np.arange(3)
plt.bar(x, height= [1,2,3])
plt.xticks(x+.5, ['a','b','c'])
plt.show()

