import numpy as np

size = 16
size_x = int(np.floor(np.sqrt(size)))
size_y = int(size // size_x)
x = np.arange(size).reshape((size_x, size_y))
print(x)

"""
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]


           top    bot   left   right
1  [0, 1] (True, False, False, False)
4  [1, 0] (False, False, True, False)
8  [2, 0] (False, False, True, False)
2  [0, 2] (True, False, False, False)                              
6  [1, 2] (False, False, False, False)                             
12 [3, 0] (False, True, True, False)                              
5  [1, 1] (False, False, False, False)                             
9  [2, 1] (False, False, False, False)                             
3  [0, 3] (True, False, False, True)                               
10 [2, 2] (False, False, False, False)                            
7  [1, 3] (False, False, False, True)                              
13 [3, 1] (False, True, False, False)                             
14 [3, 2] (False, True, False, False)                             
11 [2, 3] (False, False, False, True)                             
15 [3, 3] (False, True, False, True)                              
0  [0, 0] (True, False, True, False)      

"""
