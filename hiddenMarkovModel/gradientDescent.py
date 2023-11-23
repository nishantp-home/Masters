# Minimizing function f(w1, w2) = w1^2 + w2^4 
# using gradient descent method

maxIter = 100


# Different learning rates alpha for w1 and w2
alpha = 0.25
alpha2 = 0.25

# Initial values to assign to w1 and w2
w1i = 1.0
w2i = 1.0

# initialize w1 and w2
w1 = w1i
w2 = w2i

# Gradient descent 
for i in range(maxIter):
        w1_buffer = w1 - alpha*(2*w1)
        w2_buffer = w2 - alpha2*(4*pow(w2,3))
        w1 = w1_buffer
        w2 = w2_buffer
        print("w1:", w1)
        print("w2:", w2, "\n")

