import numpy as np
import chainer.links as L
import chainer.functions as F

print("START!")

weight = np.array([[1, 1]])
bias = -1.0
layer1 = L.Linear(2, 1, initialW=weight, initial_bias=bias)
af1 = F.relu

def AND(x):
    return af1(layer1(x))

in_data = np.array([[1,1],[0,1],[1,0],[0,0]], dtype=np.float32)
out_data = AND(in_data).data

for i in range(0, len(in_data)):
    print(in_data[i], '->', out_data[i])

print('DONE!')