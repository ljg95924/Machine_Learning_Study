# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:46:01 2018

@author: ljg
"""

import mglearn
import numpy as np
x=np.array([[1,2,3],[4,5,6]])
print('x:\n{}'.format(x))

from scipy import sparse
eye=np.eye(4)
print('NumPy array:\n{}'.format(eye))

sparse_matrix=sparse.csr_matrix(eye)
print('\nSciPy sparse CSR matrix:\n{}'.format(sparse_matrix))

data=np.ones(4)
row_indices=np.arange(4)
col_indices=np.arange(4)
eye_coo=sparse.coo_matrix((data,(row_indices,col_indices)))
print('COO representation:\n{}'.format(eye_coo))

