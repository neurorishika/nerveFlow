
# coding: utf-8

# In[2]:


from subprocess import call
import numpy as np

total_time = int(input("Enter Simulation Length ( Default: 2000 ) : ") or "2000")
n_splits = int(input("Enter Number of Splits ( Default: 2 ) : ") or "2")
time = np.split(np.arange(0,total_time,0.01),n_splits)

for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)

for i in range(n_splits):
    call(['python','run.py',str(i)])

