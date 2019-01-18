
# coding: utf-8

# In[2]:


from subprocess import call
import numpy as np

total_time = 2000
n_splits = 4
time = np.split(np.arange(0,total_time,0.01),n_splits)

for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)

for i in range(n_splits):
    call(['python','run.py',str(i)])

