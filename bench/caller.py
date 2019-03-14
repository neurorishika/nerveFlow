
# coding: utf-8

# In[2]:


from subprocess import call
import numpy as np
import sys
import time as kt

nns = [12,24,48,96,120,240,360,480]
times = [1000.0,2000.0,2500.0,3000.0,3500.0,4000.0]
rep = 5

bench = np.zeros((len(nns),len(times),rep))
eps= 0.01

for ni,n in enumerate(nns):
    for nj,t in enumerate(times):
        for r in range(rep):
            
            start = kt.time()
            call(['python','gen_input.py',str(n),str(t),str(eps)])
            
            time = np.split(np.arange(0,t,eps),int(t/100))
            
            for nt,i in enumerate(time):
                if nt>0:
                    time[nt] = np.append(i[0]-0.01,i)
            
            np.save("time",time)

            for i in range(int(t/100)):
                call(['python','run.py',str(i),str(n),str(eps)])

            bench[ni,nj,r]= kt.time()-start
            call(['rm','*.npy'])
            np.save("bench",bench)

