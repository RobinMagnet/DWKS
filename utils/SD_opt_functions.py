import numpy as np

from pyFM.optimize.base_functions import *


def SD_energy_func_std(C, descr_mu, lap_mu, descr_comm_mu, SD_comm_mu, orient_mu, descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff, SD_list):
    """
    Evaluation of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (K2*K1) or (K2,K1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    lap_mu          : scaling of the laplacian commutativity term
    descr_comm_mu   : scaling of the descriptor commutativity term
    orient_mu       : scaling of the orientation preservation term
    descr1          : (K1,p) descriptors on first basis
    descr2          : (K2,p) descriptros on second basis
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ------------------------
    energy : float - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    energy = 0

    if descr_mu > 0:
        energy += descr_mu * descr_preservation(C, descr1_red, descr2_red)

    if lap_mu > 0:
        energy += lap_mu * LB_commutation(C, ev_sqdiff)

    if descr_comm_mu > 0:
        energy += descr_comm_mu * oplist_commutation(C, list_descr)

    if orient_mu > 0:
        energy += orient_mu * oplist_commutation(C, orient_op)

    if SD_comm_mu > 0:
        energy += SD_comm_mu * oplist_commutation(C, SD_list)

    return energy


def grad_energy_std(C, descr_mu, lap_mu, descr_comm_mu, SD_comm_mu, orient_mu, descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff, SD_list):
    """
    Evaluation of the gradient of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (K2*K1) or (K2,K1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    lap_mu          : scaling of the laplacian commutativity term
    descr_comm_mu   : scaling of the descriptor commutativity term
    orient_mu       : scaling of the orientation preservation term
    descr1          : (K1,p) descriptors on first basis
    descr2          : (K2,p) descriptros on second basis
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ------------------------
    gradient : (K2*K1) - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    gradient = np.zeros_like(C)

    if descr_mu > 0:
        gradient += descr_mu * descr_preservation_grad(C, descr1_red, descr2_red)

    if lap_mu > 0:
        gradient += lap_mu * LB_commutation_grad(C, ev_sqdiff)

    if descr_comm_mu > 0:
        gradient += descr_comm_mu * oplist_commutation_grad(C, list_descr)

    if orient_mu > 0:
        gradient += orient_mu * oplist_commutation_grad(C, orient_op)

    if SD_comm_mu > 0:
        gradient += SD_comm_mu * oplist_commutation_grad(C, SD_list)

    gradient[:,0] = 0
    return gradient.reshape(-1)