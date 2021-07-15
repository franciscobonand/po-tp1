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
    # adds loose variables before 'b' column
    return np.insert(restr, [len(restr)], id_matrix, axis=1)
