import numpy as np


list1 = [1, 2, 3]

print(list1)

npArray1 = np.array(list1)
print(npArray1)

npZeros = np.zeros(7)
print(npZeros)

npOnes = np.ones(10)
print(npOnes)

npRange = np.arange(2, 11)
print(npRange)

npSteppedRange = np.arange(2, 11, 3)
print(npSteppedRange)

linearSpace = np.linspace(0, 1, 101)
print(linearSpace)
print(len(linearSpace))



anArray = np.arange(11)
sliceArray = anArray[3:6]
sliceArray2 = anArray[3::2]
sliceArray3 = anArray[3::-1]
lastArray = anArray[-5::2]
print(anArray)
print(sliceArray)
print(sliceArray2)
print(sliceArray3)
print(lastArray)
print(np.amin(anArray), np.amax(anArray), np.average(anArray))

an_array = np.arange(0, 6)
print(an_array)

an_array[0] = 10

print(np.append(an_array, [12]))
print(np.insert(an_array, 1, 111))
print(an_array)
print(np.delete(an_array, 4))
print(np.sort(an_array))