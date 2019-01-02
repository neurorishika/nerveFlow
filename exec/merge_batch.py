import numpy as np

batch = int(input("Enter Number of Batches:"))

for n,i in enumerate(["state.batch"+str(n+1)+".npy" for n in range(batch)]):
    if n<1:
        l=np.load(i)
    else:
        l = np.concatenate([l,np.load(i)[1:,:]])

np.savetxt("state.vector",l,delimiter=",",fmt="%.3f")
