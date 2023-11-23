import numpy as np
import theano
import theano.tensor as T

# Define a scalar N
N = T.iscalar('N')


def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1

outputs, updates = theano.scan(
    fn=recurrence,
    sequences=T.arange(N),   #python arange function
    n_steps=N,
    outputs_info=[1., 1.]   #Theano expects a list here
)

fibonacci = theano.function(
    inputs=[N],
    outputs=outputs
)

o_val = fibonacci(8)

print("output:", o_val)