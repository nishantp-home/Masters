import numpy as np

# Rank 1 array
array1 = [2,3,4,5]
print("Array 1 type:", type(array1))
numpyArray1 = np.array(array1)
print("Numpy Array1 type:", type(numpyArray1))

# Multi-dimensional  array
array2 = [[1, 2, 3], [4, 5, 6]]
numpyArray2 = np.array(array2)
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