import numpy as np

# Rank 1 array
array1 = [2,3,4,5,6,7,8]
print("Array 1 type:", type(array1))
numpyArray1 = np.array(array1)
print("Numpy Array1 type:", type(numpyArray1))
print(array1[-1]) # Last item of the list 
print(array1[-4:-1])  # Negative indices
print(array1[1:4:2])  # Step of 2
print(numpyArray1[-1]) # Last item of the list 
print(numpyArray1[-4:-1])  # Negative indices
print(numpyArray1[1:4:2])  # Step of 2

array1_copy = numpyArray1[:]  # Making a copy of an array
print(array1_copy)

# Concatenate two arrays
numpyArray2 = numpyArray1[1:3]
print("numpyArray1:", numpyArray1)
print("numpyArray2:", numpyArray2)
concatenatedArray = np.concatenate([numpyArray1, numpyArray2])
print("concatenatedArray:", concatenatedArray)

# Multi-dimensional  array
array2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Array 2 type:", type(array2))
numpyArray2 = np.array(array2)
print("Row at index 1 :", numpyArray2[1])
print("Column at index 1 :", numpyArray2[:, 1])

# Concatenate 2D np arrays 
concatenated2DArray = np.concatenate((numpyArray2, np.array([[10, 11, 12]])))
rightConcatenated2DArray = np.concatenate((numpyArray2, np.array([[10], [11], [12]])), axis=1)
print("concatenated2DArray:", concatenated2DArray)
print("rightConcatenated2DArray:", rightConcatenated2DArray)

print("Numpy array 2 type:", type(numpyArray2))
print("Array2:", array2)
print("Numpy array2:", numpyArray2)
print("Shape of numpy array 2:", numpyArray2.shape)

# Creating different types of Numpy arrays
print("2*2 Null matrix:", np.zeros((2,2)))    # Null matrix of size 2*2
print("2*2 matrix with 5:", np.full((2,2), 5))   # 2*2 array with 5
print("2*2 Identity matrix:", np.eye(2,2))     # Identity matrix of size 2*2 

# Operations on numpy arrays
array = [[1,2,3,4],[3,4,5,6],[2,5,7,9]]
numpyArray = np.array(array)
print("A numpy array:", "\n", numpyArray)
print("Sum of all entries:", numpyArray.sum())   
print("Column-wise sum:", numpyArray.sum(axis=0)) 
print("Row-wise sum:", numpyArray.sum(axis=1))  
print("Column-wise mean:", numpyArray.mean(axis=0))  
print("Row-wise mean:", numpyArray.mean(axis=1))  
print("Row-wise median:", np.median(numpyArray, axis=1))   
print("Row-wise standard deviation", np.std(numpyArray, axis=1))
print("50-Percentile value:", np.percentile(numpyArray, 50, axis=1))


# Vector Arithmatic
x = np.array([1,2,3])
y = np.array([2,3,4])

print('x+y:', x+y)  # vector addition
print('x+2:', x+2)  # scalar addition to a vector
print('Hadamard product x*y:', x*y)  # Hadamard product (element wise product)
print('Dot product x.y:', x.dot(y))  # np.dot(x, y) can also be used
print('Element-wise division x/y:', y/x)


# Matrix arithmatic
x = np.matrix([[1,2], [3,4]])
y = np.matrix([[5,3], [8,7]])
# Numpy matriics are strictly 2-dimensional, while numpy arrays (ndarrays) are N-dimensional.
# Numpy matrix provides a convenient notation for matrix multiplication
print(type(y))
print('x+y:', x+y)
print('Transpose of x:', x.T)
print('Product x*y:\n', x*y) # Matrix multiplication
print('Hadamard product x*y:\n', np.multiply(x, y))  # Element-wise product
print("Inverse of x:", np.linalg.inv(x))

# Broadcasting
A = np.arange(20).reshape(5,4)
print("A:\n", A)
B = np.arange(5).reshape(5,1)
print("B:\n", B)
print("A+B:\n", A+B)
print("A*B:\n", A*B)

# Solve linear equations AX = b -> X = inv(A)*b, X = [x1,x2,x3]
A = np.array([[1,2,3],[4,5,2],[2,8,5]])
b = np.array([5,10,15])
x= np.linalg.solve(A,b)
print(x)