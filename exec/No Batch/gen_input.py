import numpy as np
import matplotlib.pyplot as plt

n_n = int(input("Number of Neurons (Default = 120) : ") or "120")
time = float(input("Time in ms (Default = 1000) : ") or "1000")
eps = float(input("Time Resolution in ms (Default = 0.01) : ") or "0.01")

current_input = np.zeros((n_n,int(time/eps)))
t = np.arange(0,time,eps)

y = np.where(t<600,(1-np.exp(-(t-100)/75)),0.9996*np.exp(-(t-600)/150))
y = np.where(t<100,np.zeros(t.shape),y)

fac = 10
p_input = 0.33
input_neurons = np.random.choice(np.array(range(n_n)),int(p_input*n_n),replace=False)

current_input[input_neurons,:]= fac*y

np.save("current",current_input)
