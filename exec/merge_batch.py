import numpy as np

batch = int(input("Enter Number of Batches:"))

for n,i in enumerate(["state.batch"+str(n+1)+".npy" for n in range(batch)]):
    if n<1:
        l=np.load(i)
    else:
        l = np.concatenate([l,np.load(i)[1:,:]])

print(l.shape)

output = input("Binary(B) or ASCII(A):")

if output == "B":
    np.save("output",l)
elif output == "A":
    np.savetxt("output.csv",l,delimiter=",",fmt="%.3f")
else:
    print("Wrong Input. Saving as Binary.")
    np.save("output",l)
