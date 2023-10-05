import numpy as np

twoByTwo = np.matrix([[1,2],[3,4]])
print(twoByTwo)
twoByThree = np.matrix('1,2; 3,4; 5,6')
print(twoByThree)

print(np.zeros([4,3]))
print(np.ones([4,3]))

aMatrix = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print(aMatrix)
print(aMatrix[0])
print(aMatrix[1,2])
print(aMatrix.shape, aMatrix.size)
print(np.average(aMatrix), aMatrix.max())


new_matrix = aMatrix
new_matrix[2,0] = 10
print(new_matrix)
print(np.sort(new_matrix))
print(new_matrix.T)
print(new_matrix.flatten())