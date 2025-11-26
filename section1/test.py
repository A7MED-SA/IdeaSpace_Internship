import numpy as np

arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr2 = np.array([[[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]]])
arr3 = np.array([[1,2,3],[4,5,6]])

print(arr)
print(arr.shape)
print(type(arr))
print(arr.dtype)

print(arr2)
print(arr2.shape)
print(type(arr2))
print(arr2.dtype)

print(arr3)
print(arr3.shape)
print(type(arr3))
print(arr3.dtype)
