import copy

import scipy.sparse as sparse
import numpy as np

from sklearn.neighbors import KDTree

import pyFM.spectral as spectral


def get_SD_list(collection, p2p=None, k1=50):
    SD_list = []
    meshA = collection[0]
    for indB, meshB in enumerate(collection):
        if p2p is not None:
            map_B = p2p[indB]
        else:
            map_B = None
        SD_AB_a, SD_AB_c = spectral.compute_SD(meshA, meshB, k1=k1, p2p=map_B, SD_type='spectral')  # (k1,k1), (k1,k1)

        SD_list.append([SD_AB_a, SD_AB_c])

    return SD_list


def get_sparse_commut(A,B):
    """
    Returns matrix C so that C@X.ravel() = (A@X - X@B).ravel()

    Parameters
    -----------------------
    A : (k2,k2) array
    B : (k1,k1) array

    Output
    -----------------------
    C : (k1*k2,k1*k2) sparse
    """
    k1 = B.shape[0]
    k2 = A.shape[1]

    # Term C1 so that C1@X.ravel() = A@X
    term2 = sparse.block_diag([sparse.csc_matrix(B.T) for _ in range(k2)], format='csc')

    # Term C2 so that C2@X.ravel() = X@B
    term1 = sparse.block_diag([sparse.csc_matrix(A) for _ in range(k1)], format='csc')
    reindex = np.arange(k1*k2).reshape((k2, k1), order='F').ravel()
    term1 = term1[reindex[:, None], reindex]

    return term1 - term2


def solve_collection(SD1_list, SD2_list, meshA, meshC, alpha=.1, remove_area=False, remove_conformal=False):
    """
    Solve the least square problem using SVD
    """
    k1, k2 = SD1_list[0][0].shape[0], SD2_list[0][0].shape[0]
    n_meshes = len(SD1_list)

    print('Building Least Square Matrix')
    C = sparse.csc_matrix((k1*k2, k1*k2))
    for i in range(n_meshes-1):

        if not remove_area:
            term_a = get_sparse_commut(SD2_list[i][0], SD1_list[i][0])
            C += copy.deepcopy(term_a.T @ term_a)

        if not remove_conformal:
            term_c = get_sparse_commut(SD2_list[i][1], SD1_list[i][1])
            C += copy.deepcopy(term_c.T @ term_c)

    L = get_sparse_commut(sparse.diags(meshC.eigenvalues[:k2]), sparse.diags(meshA.eigenvalues[:k1]))

    # Normalize C and L before applying regularization
    X_test = np.eye(k2,k1)
    beta = (X_test.ravel()@(C@X_test.ravel())) / np.linalg.norm(L@X_test.ravel())**2

    print(f'\tRescaling alpha by {beta:.1e}')
    print('Computing rank :')
    rank = np.linalg.matrix_rank((C+alpha*beta*copy.deepcopy(L.T@L)).todense())
    print(f'\trank {rank:d}/{C.shape[0]}')
    print('Solving system')
    eigenvalues, eigenvectors = sparse.linalg.eigsh(C+alpha*beta*copy.deepcopy(L.T@L), which='LM',sigma=-1e-8,k=1+2500-rank)
    print(f'\tSmallest eigenvalue : {eigenvalues[-1]:.3f}')

    myFM = eigenvectors[:,-1].reshape((k2, k1))

    # The solution is defined up a a sign flip
    myp2p = spectral.FM_to_p2p(myFM, meshA.eigenvectors, meshC.eigenvectors)
    myp2p2 = spectral.FM_to_p2p(-myFM, meshA.eigenvectors, meshC.eigenvectors)

    # The best solution is the one with the smallest embedding distances
    tree = KDTree((myFM @ meshA.eigenvectors[:,:k1].T).T)  # Tree on (n1,k2)
    dists1,_ = tree.query(meshC.eigenvectors[:,:k2],k=1,return_distance=True)

    tree = KDTree((-myFM@meshA.eigenvectors[:,:k1].T).T)  # Tree on (n1,k2)
    dists2,_ = tree.query(meshC.eigenvectors[:,:k2],k=1,return_distance=True)

    p2p_comm = myp2p if dists1.mean() < dists2.mean() else myp2p2
    FM_comm = myFM if dists1.mean() < dists2.mean() else -myFM

    return FM_comm, p2p_comm
