import numpy as np

list1=[]
nplist=np.array([])
a=1
b=2
list1.append(a)
list1.append(b)

nplist=np.append(nplist,a)
nplist=np.append(nplist,b)

print(np.array(list1))
print(nplist)