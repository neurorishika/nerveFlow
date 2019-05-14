import numpy as np
import matplotlib.pyplot as plt
import sys

n_n = int(sys.argv[1])
time = float(sys.argv[2])
eps = float(sys.argv[3])

current_input = np.zeros((n_n,int(time/eps)))
t = np.arange(0,time,eps)

y = np.tile(np.concatenate([np.zeros(int(500/eps)),np.ones(int(500/eps))],axis=0),int(time/1000)+1)[:t.shape[0]]

fac = 10
p_input = 0.33
input_neurons = np.random.choice(np.array(range(n_n)),int(p_input*n_n),replace=False)

current_input[input_neurons,:]= fac*y

np.save("current",current_input)
