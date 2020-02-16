import numpy as np

# AHP Module.
RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90,
           5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def AHP_Check(array):
    '''
    Check if the matrix is available for AHP.
    '''
    row = array.shape[0]
    a_axis_0_sum = array.sum(axis=0)
    b = array / a_axis_0_sum
    b_axis_1_sum = b.sum(axis=1)
    w = b_axis_1_sum / row
    AW = (w * array).sum(axis=1)
    max_max = sum(AW / (row * w))
    CI = (max_max - row) / (row - 1)
    CR = CI / RI_dict[row]
    if CR < 0.1:
        return True, w
    else:
        return False, w
