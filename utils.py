import numpy as np


def create_VERO(c_T, restr):
    aux_table_row1 = [0] * len(restr)
    c_T_zeroes = [0] * (len(restr) + 1)
    # creates first row of VERO
    head = np.append(aux_table_row1, c_T)
    head = np.negative(np.append(head, c_T_zeroes))
    # creates identity matrix for VERO's auxiliar table
    aux_table = np.identity(len(restr))
    # puts restrictions in FPI
    restr_FPI = format_FPI(restr)
    # generate tableaux but it's first row
    body = np.append(aux_table, restr_FPI, axis=1)
    return np.insert(body, 0, head, axis=0)


def format_FPI(restr):
    id_matrix = np.identity(len(restr))
    b = np.squeeze(np.asarray(restr[:, restr.shape[1] - 1]))
    wout_b = np.delete(restr, restr.shape[1]-1, 1)
    # adds loose variables before 'b' column
    # fpi = np.insert(id_matrix, [len(wout_b)], wout_b, axis=1)
    fpi = np.append(wout_b, id_matrix, axis=1)
    return np.c_[fpi, b]


def check_and_resolve_negative_b(vero):
    b = np.squeeze(np.asarray(vero[1:, vero.shape[1] - 1]))
    for i in range(0, len(b)):
        if b[i] < 0:
            vero[i+1] = np.negative(vero[i+1])


def create_aux_lp(restr):
    id_matrix = np.identity(restr.shape[0])
    head = np.append([0] * (restr.shape[1] - 1), [-1] * (restr.shape[0]))
    head = np.append(head, [0])
    b = np.squeeze(np.asarray(restr[:, restr.shape[1] - 1]))
    wout_b = np.delete(restr, restr.shape[1]-1, 1)
    fpi = np.append(wout_b, id_matrix, axis=1)
    fpi = np.c_[fpi, b]
    return np.insert(fpi, 0, head, axis=0)
