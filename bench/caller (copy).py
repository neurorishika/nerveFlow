
# coding: utf-8

# In[2]:


from subprocess import call
import numpy as np
import sys

total_time = float(sys.argv[1])
n_splits = int(sys.argv[2])
n_n = int(sys.argv[3])
eps = float(sys.argv[4])

call(['python','gen_input.py',str(n_n),str(total_time),str(eps)])

time = np.split(np.arange(0,total_time,eps),n_splits)

for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)

for i in range(n_splits):
    call(['python','run.py',str(i),str(n_n),str(eps)])

#print("Simulation Completed. Time taken: {:0.2f}".format(t.time()-start))
