import numpy as np
from collections import OrderedDict
from scipy.optimize import linprog


def sort_and_extract_keys(compared2best, compared2worst):
    cb = OrderedDict(sorted(compared2best.items()))
    cw = OrderedDict(sorted(compared2worst.items()))
    allkeys = list(cb.keys())
    return cb, cw, allkeys


def create_best_criterion_matrices(cb, allkeys, bkey):
    cb_copy = cb.copy()
    cb_copy.pop(bkey, None)
    colSize = len(allkeys)
    tmpmat = np.zeros((len(cb_copy), colSize + 1), dtype=np.double)
    tmpmat1 = np.zeros((len(cb_copy), colSize + 1), dtype=np.double)

    for idx, key in enumerate(cb_copy):
        itmp = allkeys.index(key)
        tmpmat[idx, allkeys.index(bkey)] = 1.0
        tmpmat[idx, itmp] = -cb_copy[key]
        tmpmat[idx, colSize] = -1.0

        tmpmat1[idx, allkeys.index(bkey)] = -1.0
        tmpmat1[idx, itmp] = cb_copy[key]
        tmpmat1[idx, colSize] = -1.0

    return np.concatenate((tmpmat, tmpmat1), axis=0)


def create_worst_criterion_matrices(cw, allkeys, bkey, wkey):
    cw_copy = cw.copy()
    cw_copy.pop(bkey, None)
    cw_copy.pop(wkey, None)
    colSize = len(allkeys)
    tmpmat = np.zeros((len(cw_copy), colSize + 1), dtype=np.double)
    tmpmat1 = np.zeros((len(cw_copy), colSize + 1), dtype=np.double)

    for idx, key in enumerate(cw_copy):
        itmp = allkeys.index(key)
        tmpmat[idx, itmp] = 1
        tmpmat[idx, allkeys.index(wkey)] = -cw_copy[key]
        tmpmat[idx, colSize] = -1.0

        tmpmat1[idx, itmp] = -1
        tmpmat1[idx, allkeys.index(wkey)] = cw_copy[key]
        tmpmat1[idx, colSize] = -1.0

    return np.concatenate((tmpmat, tmpmat1), axis=0)


def solve_linear_program(mat, colSize):
    rowSize = mat.shape[0]
    Aeq = np.ones((1, colSize + 1), dtype=np.double)
    Aeq[0, -1] = 0.0
    beq = np.array([1])
    bub = np.zeros((rowSize), dtype=np.double)
    cc = np.zeros((colSize + 1), dtype=np.double)
    cc[-1] = 1

    res = linprog(
        cc,
        A_eq=Aeq,
        b_eq=beq,
        A_ub=mat,
        b_ub=bub,
        bounds=(0, None),
        options={"disp": False},
    )
    return res["x"]


def calculate_weight(compared2best, compared2worst, bkey, wkey):
    cb, cw, allkeys = sort_and_extract_keys(compared2best, compared2worst)
    colSize = len(allkeys)
    rowSize = 4 * colSize - 5

    mat = np.zeros((rowSize - 1, colSize + 1), dtype=np.double)
    best_crit_matrices = create_best_criterion_matrices(cb, allkeys, bkey)
    worst_crit_matrices = create_worst_criterion_matrices(cw, allkeys, bkey, wkey)

    mat[: 2 * colSize - 2, :] = best_crit_matrices
    mat[2 * colSize - 2 :, :] = worst_crit_matrices

    sol1 = solve_linear_program(mat, colSize)

    outp = {key: sol1[idx].item() for idx, key in enumerate(allkeys)}
    return outp
