import numpy as np


# Softmax calculation for one sample
# Assume that a is the activation at five different nodes
a = np.random.randn(5)
expa =np.exp(a)
answer = expa/expa.sum()


# Softmax calculation for 100 sample
A = np.random.randn(100, 5)
expA = np.exp(A)
Answer = expA/expA.sum(axis=1, keepdims=True)  # sum along rows, keepdims=True for dividing 
                                               # each row of matrix by the corresponding element of the vector 