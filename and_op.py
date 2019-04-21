
import numpy as np
import chainer.links as L
import chainer.functions as F

print("START!")

W = np.array([[1, 1]])
b = -1.0 # bias
p = L.Linear(2, 1, initialW=W, initial_bias=b)
f = F.relu

def AND(x):
    return f(p(x))

x = np.array([[1,1],[0,1],[1,0],[0,0]], dtype=np.float32) # input data
print(AND(x).data)

print('DONE!')