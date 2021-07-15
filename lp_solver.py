import numpy as np
import sys
import utils

try:
    matrix_size = list(map(lambda x: int(x), input().strip().split(" ")))
    c_T = np.array(list(map(lambda x: int(x), input().strip().split(" "))))
    rest_str = ""
    for i in range(0, matrix_size[0]):
        rest_str += input().strip()
        if i < matrix_size[0]-1:
            rest_str += ";"
    restrictions = np.matrix(rest_str)
except:
    print("invalid input arguments")
    sys.exit(1)

vero = utils.create_VERO(c_T, restrictions)
print(vero)
auxT_end = len(restrictions) - 1
tab_init = len(restrictions)
tab_end = np.size(vero, 1) - 2
b = np.size(vero, 1) - 1
print(auxT_end, tab_init, tab_end, b)
