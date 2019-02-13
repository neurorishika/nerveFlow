from subprocess import call, Popen, PIPE
import time as kt
import numpy as np

nns = [12,24]#,48,96,120,240,360,480]
time = [1000.0,1500.0]#,2000.0,2500.0,3000.0,3500.0,4000.0]
rep = 1

bench = np.zeros((len(nns),len(time),rep))

for i,n in enumerate(nns):
    for j,t in enumerate(time):
        for r in range(rep):
            start = kt.time()
            call(['python','caller.py',str(t),str(int(t/500)),str(n),str(0.01)])
            bench[i,j,r]= kt.time()-start

np.savetxt("bench.txt",bench)
