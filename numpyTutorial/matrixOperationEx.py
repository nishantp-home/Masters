import numpy as np

imageMatrix = np.matrix([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
print(imageMatrix)

def invertImage(imageMatrix):
    rows = imageMatrix.shape[0]
    columns = imageMatrix.shape[1]
    for row in range(rows):
        for column in range(columns):
            if(imageMatrix[row, column] == 0):
                imageMatrix[row, column] = 1
            else:
                imageMatrix[row, column] = 0
    return imageMatrix
    

print(invertImage(imageMatrix))
