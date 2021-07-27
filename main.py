import numpy as np
import sys
from numpy import round
import utils

try:
    n_restr = n_vars = 0
    c_T = []
    restrictions = []
    b = []
    for idx, line in enumerate(sys.stdin):
        if idx == 0:
            n_restr, n_vars = map(int, line.split())
        elif idx == 1:
            c_T = np.array(list(map(int, line.split())))
        else:
            cur_line = list(map(int, line.split()))
            restrictions += [np.array(cur_line[:-1])]
            b += [cur_line[-1]]

except:
    print("invalid input arguments")
    sys.exit(1)

vero = utils.create_VERO(c_T, np.matrix(restrictions), b)
tabl = utils.return_vero_wout_auxtable(vero, n_restr)

# total_vars represents the inputed variables + loose variables
total_vars = n_vars + n_restr
valid_base = utils.get_base(tabl, n_restr, total_vars)

result, opt_val, opt_base, cert = utils.solver(
    vero, n_restr, total_vars, total_vars, valid_base)

print(result)

if result == "ilimitada":
    last_cert = total_vars - n_restr
else:
    last_cert = n_restr

if opt_val is not None:
    print(f"{round(opt_val, 7):.7f}")

if opt_base is not None:
    for i in opt_base[:total_vars - n_restr]:
        print(f"{round(i, 7):.7f}", end=' ')
    print()

if cert is not None:
    for i in cert[:last_cert]:
        print(f"{round(i, 7):.7f}", end=' ')
    print()
