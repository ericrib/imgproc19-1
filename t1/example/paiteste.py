import numpy as np
a = np.random.randint(0,10,(10,20))
print(a)
iterzinho = np.nditer(a,flags=['multi_index'],op_flags=['readwrite'])

with iterzinho:
    while not iterzinho.finished:
        iterzinho[0] = iterzinho.multi_index[0]*100 + iterzinho.multi_index[1]
        print(iterzinho[0])
        print(type(iterzinho))
        iterzinho.iternext()


print(a)