import tensorflow as tf

a = tf.constant(6., name='constant_a')
b = tf.constant(3., name='constant_b')
c = tf.constant(10., name='constant_c')
d = tf.constant(5., name='constant_d')

# a, b, c are tensors flowing through the graph
mul = tf.multiply(a, b, name='mul')      # mul is a node which specifies multiply operation on 'a' and 'b'
div = tf.divide(c, d, name='div')        # division operation which divides element wise c by d and have the name 'div'
addn = tf.add_n([mul, div], name='addn')
print("mul:", mul)
print("div:", div)
print("addn:", addn)



# Constants and variables

# Constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

print("x1 * x2 = ", tf.multiply(x1, x2))   # multiplication 


# Variables
x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')
print(y.numpy()) # type: ignore


# Placeholders in tensorflow

# To prevent the following runtime error:
# 'tf.placeholder() is not compatible with eager execution'
tf.compat.v1.disable_eager_execution()   # tf.compat.v1 is the compatible api version with the specific methods


# Example 1
# setup placeholders
x = tf.compat.v1.placeholder(tf.int32, shape=[3], name='x')
y = tf.compat.v1.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')

final_mean=tf.reduce_mean([sum_x, prod_y], name='final_mean')
sess = tf.compat.v1.Session()
print("sum(x)::", sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print("prod(y):", sess.run(prod_y, feed_dict={y: [1,2,3]}))
writer = tf.compat.v1.summary.FileWriter('tensorFlow_example', sess.graph)


# Example 2
x = tf.compat.v1.placeholder("float", [None, 3])
y = x * 2

with tf.compat.v1.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)