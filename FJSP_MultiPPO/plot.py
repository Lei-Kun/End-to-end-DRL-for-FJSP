
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Params import configs



a = np.loadtxt('./N_%s_M%s_U24'%(configs.n_j,configs.n_m),delimiter="\n")


a = a[0:400]

time = range(a.shape[0])

sns.set(style="darkgrid", font_scale=1)
sns.tsplot(time=time, data=a, color="r")

plt.ylabel("Average Makespan")

plt.xlabel("Iteration Number")
#plt.title("Imitation Learning")
plt.savefig("./N_%s_M%s.png"%(configs.n_j,configs.n_m), dpi=500, bbox_inches='tight')

plt.show()