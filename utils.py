import numpy as np


def create_VERO(c_T, restr, b):
    aux_table_row1 = [0] * restr.shape[0]
    c_T_zeroes = [0] * (restr.shape[0] + 1)
    # creates first row of VERO
    head = np.append(aux_table_row1, c_T)
    head = np.negative(np.append(head, c_T_zeroes))
    # creates identity matrix for VERO's auxiliar table
    # and loose restrictions
    id_matrix = np.identity(restr.shape[0])
    # appends identity in both sides of restrictions
    body = np.append(id_matrix, restr, axis=1)
    body = np.append(body, id_matrix, axis=1)
    # appends b column as the last column
    body = np.c_[body, np.matrix(b).T]

    return np.asarray(np.insert(body, 0, head, axis=0))


def get_base(tabl, n_restr, n_vars):
    c_T = tabl[0, :-1]
    restr = tabl[1:, :-1]
    base = np.zeros(n_restr)
    current = 0
    for i in range(0, n_vars):
        if c_T[i] == 0 and restr[restr[:, i] == 1, current].shape[0] == 1:
            base[current] = i
            current += 1
            if current == n_restr:
                break

    return base


def solver(vero, n_restr, n_vars, n_costs, b):
    # generates auxiliar linear programming
    lp_aux = create_aux_lp(vero[1:, n_restr:])
    lp_aux[0] -= lp_aux[1:].sum(axis=0)
    # solves auxiliar linear programming with simplex
    _, ot, __, cert, lp_aux, b = simplex(lp_aux, n_restr, n_vars +
                                         n_restr, n_vars + n_restr, np.arange(n_restr) + n_vars)

    # checks if optimum value of auxiliar lp is negative
    if ot is not None and np.round(ot, 7) < 0:
        return "inviavel", None, None, cert

    tabl = return_vero_wout_auxtable(vero, n_restr)
    # updates vero table with bases extracted from solving auxiliar lp
    vero[:, :n_restr + n_vars] = lp_aux[:, :n_restr + n_vars]
    vero[0, n_restr:] = tabl[0] + lp_aux[0, :n_restr].T@tabl[1:]
    vero[:, -1] = lp_aux[:, -1]

    for i in b:
        pivot(vero, i + n_restr, np.arange(n_restr)
              [vero[1:,  n_restr + i] != 0][0] + 1)

    return simplex(vero, n_restr, n_vars, n_costs, b)[:-2]


def simplex(vero, n_restr, n_vars, n_costs, b):
    while True:
        tabl = return_vero_wout_auxtable(vero, n_restr)

        # checks if all elements in c_T are greater than zero
        if (-tabl[0, :-1] <= 0).all():
            return "otima", tabl[0, -1], optim_base(tabl, n_vars, b),  vero[0, :n_restr], vero, b

        # gets index of next base candidate
        nxt_base_candidate = (np.arange(n_costs)[tabl[0, :-1] < 0])[0]
        # checks if all elements of selected row are negative
        if (tabl[1:, nxt_base_candidate] <= 0).all():
            cert = np.zeros(n_vars)
            cert[b[b < n_vars]] = tabl[1:, nxt_base_candidate]
            cert *= -1
            cert[nxt_base_candidate] = 1
            return "ilimitada", None,  optim_base(tabl, n_vars, b), cert[:n_vars], vero, b

        row = tabl[1:, nxt_base_candidate]
        indices = np.arange(row.shape[0])[row > 0] + 1
        index = indices[np.argmin((tabl[indices, -1]/row[indices - 1]))]
        current = tabl[index][b] != 0
        b[current] = nxt_base_candidate

        pivot(vero, nxt_base_candidate + n_restr, index)


# Auxiliar functions


def pivot(tabl, col, row):
    tabl[row] /= tabl[row, col]

    tabl[:row] -= (tabl[:row, col] / tabl[row, col]
                   )[:, np.newaxis] * tabl[row, ]

    tabl[row + 1:] -= (tabl[row + 1:, col] / tabl[row, col]
                       )[:, np.newaxis] * tabl[row, ]


def optim_base(tableaux, n_vars, b):
    # mask to filter valid base values
    mask = np.zeros(n_vars)
    mask[b[b < n_vars]] = 1

    base = np.zeros(n_vars)
    for row in tableaux[1:]:
        base += row[:n_vars] * mask * row[-1]

    return base


def return_vero_wout_auxtable(vero, n_restr):
    vero_cpy = vero.copy()
    return vero_cpy[:, n_restr:]


def create_aux_lp(restr):
    id_matrix = np.identity(restr.shape[0])
    head = np.append([0] * (restr.shape[1] - 1), [1] * (restr.shape[0]))
    head = np.append(head, [0])
    head = np.append([0]*restr.shape[0], head)
    fpi = np.append(id_matrix, restr, axis=1)
    # invert signals of lines that contain negative value for b
    check_and_resolve_negative_b(fpi)
    b = np.squeeze(np.asarray(fpi[:, fpi.shape[1] - 1]))
    wout_b = np.delete(fpi, fpi.shape[1]-1, 1)
    fpi = np.append(wout_b, id_matrix, axis=1)
    fpi = np.c_[fpi, b]
    return np.asarray(np.insert(fpi, 0, head, axis=0))


def check_and_resolve_negative_b(tabl):
    b = np.squeeze(np.asarray(tabl[:, tabl.shape[1] - 1]))
    for i in range(0, len(b)):
        if b[i] < 0:
            tabl[i] = np.negative(tabl[i])
