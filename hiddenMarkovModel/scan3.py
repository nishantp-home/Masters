import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

# Low pass filter

# Create a signal with (a lot of) noise, underlying trend is sin wave
X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))
plt.plot(X)
plt.title('Original')
plt.show()

#define variable
decay = T.scalar('decay')

#define vector
sequence = T.vector('sequence')


def recurrence(x, last, decay):
    return (1-decay)*x + decay*last

outputs, _ = theano.scan(
    fn=recurrence,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[decay]   # Arguments that we don't want to loop through
)

lpf = theano.function(
    inputs=[sequence, decay],
    outputs=outputs,
)

Y = lpf(X, 0.99)
plt.plot(Y)
plt.title('Filtered')
plt.show()