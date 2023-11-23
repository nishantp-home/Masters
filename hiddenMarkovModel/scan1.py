import numpy as np
import theano
import theano.tensor as T

#set x to be a vector
x = T.vector('x')


# Define a function square
def square(x):
    return x*x


#Call theano scan
outputs, updates = theano.scan(
    fn=square,
    sequences=x,
    n_steps=x.shape[0]
)

# Create a theano function
square_op = theano.function(
    inputs=[x],
    outputs=[outputs]
)

o_val = square_op(np.array([1, 2, 3, 4, 5]))

print("o_val:", o_val)