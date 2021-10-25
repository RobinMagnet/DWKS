import os
import copy
import time
import numpy as np

import scipy.linalg
from scipy.optimize import fmin_l_bfgs_b

from pyFM.mesh import TriMesh
import pyFM.signatures as sg
import pyFM.optimize as opt_func
# import pyFM.optimize.SD_functions as SD_opt
import SD_opt_functions as SD_opt
import pyFM.refine.icp as icp
import pyFM.refine.zoomout as zoomout
import pyFM.spectral as spectral


class MultiShapeDifferenceMatching:
    """
    A class enabling all functions to realise functional maps
    """

    def __init__(self, meshA, meshC):

        # Base Shapes
        self.meshA = copy.deepcopy(meshA)
        self.meshC = copy.deepcopy(meshC)

        # Complete descriptors
        self.descr1 = None
        self.descr2 = None

        # Dimension of the functional map
        self._k1 = None
        self._k2 = None

        # Type of shape differences
        self.SD_type = None

        # Intra Collection maps
        self.mapping = None

        # Pointwise map
        self.p2p = None

        self.FM_type = 'classic'
        self._FM_base = None
        self._FM_icp = None
        self._FM_zo = None

        # Shape Differences, their eigenvalues and eigenvectors
        self.SD_AB_list = []
        self.SD_CD_list = []

        self.evals_AB_list = []
        self.evals_CD_list = []

        self.evects_AB_list = []
        self.evects_CD_list = []

    @property
    def k1(self):
        if self._k1 is None and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[1]
        else:
            return self._k1

    @k1.setter
    def k1(self, k1):
        self._k1 = k1

    @property
    def k2(self):
        if self._k2 is None and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[0]
        else:
            return self._k2

    @k2.setter
    def k2(self, k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    @property
    def FM_type(self):
        return self._FM_type

    @FM_type.setter
    def FM_type(self, FM_type):
        if FM_type.lower() not in ['classic', 'icp', 'zoomout']:
            raise ValueError(f'FM_type can only be set to "classic", "icp" or "zoomout", not {FM_type}')
        self._FM_type = FM_type

    def change_FM_type(self, FM_type):
        self.FM_type = FM_type

    @property
    def FM(self):
        """
        Returns the current functional map depending on the value of FM_type

        Output
        ----------------
        FM : (k2,k1) current FM
        """
        if self.FM_type.lower() == 'classic':
            return self._FM_base
        elif self.FM_type.lower() == 'icp':
            return self._FM_icp
        elif self.FM_type.lower() == 'zoomout':
            return self._FM_zo

    @FM.setter
    def FM(self, FM):
        self._FM_base = FM

    @property
    def p2p(self):
        """
        Computes and return the current point to point map

        Output
        --------------------
        p2p : (n2,) point to point map associated to the current functional map
        """
        if not self.fitted or not self.preprocessed:
            raise ValueError('Model should be processed and fit to obtain p2p map')
        if self._p2p is None:
            self.p2p = spectral.FM_to_p2p(self.FM, self.meshA.eigenvectors, self.meshC.eigenvectors)
        return self._p2p

    @p2p.setter
    def p2p(self, p2p):
        self._p2p = p2p

    @property
    def preprocessed(self):
        test_descr = (self.descr1 is not None) and (self.descr2 is not None)
        return test_descr  # and test_evals and test_evects

    @property
    def fitted(self):
        return self.FM is not None

    def compute_SD(self, meshB, meshD, k_A=30, k_C=30, maps=None, SD_type='spectral', verbose=False):
        """
        Computes shape difference operators and their spectral properties between meshA and meshB,
        and between meshC and meshD


        Parameters
        --------------------------------
        meshB   : TriMesh - mesh in the first collection
        meshD   : TriMesh - mesh in the second collection
        k_A     : int - size of the shape difference operators to compute for collection 1
        k_C     : int - size of the shape difference operators to compute for collection 2
        maps    : tuples((n_B,), (n_D,)) with first element being the pointwise map between
                  meshA and meshB, and second element the pointwise map between meshC and meshD.
                  If not specified, meshes are assumed to be in 1 to 1 corerspondence
        SD_type : str - either 'spectral' or 'semican'.
        """
        map_B, map_D = None, None
        if maps is not None:
            map_B, map_D = maps

        # Compute shape differences
        SD_AB_a, SD_AB_c = spectral.compute_SD(self.meshA, meshB, k1=k_A, p2p=map_B,
                                               SD_type=SD_type)  # (k1,k1), (k1,k1)
        SD_CD_a, SD_CD_c = spectral.compute_SD(self.meshC, meshD, k1=k_C, p2p=map_D,
                                               SD_type=SD_type)  # (k2,k2), (k2,k2)

        # Compute spectral values
        evals_AB_a, evects_AB_a = scipy.linalg.eigh(SD_AB_a)  # (k1,), (k1,k1)
        evals_CD_a, evects_CD_a = scipy.linalg.eigh(SD_CD_a)  # (k2,), (k2,k2)
        evals_AB_c, evects_AB_c = scipy.linalg.eigh(SD_AB_c)  # (k1,), (k1,k1)
        evals_CD_c, evects_CD_c = scipy.linalg.eigh(SD_CD_c)  # (k2,), (k2,k2)

        # Save values
        self.SD_AB_list.append([SD_AB_a, SD_AB_c])
        self.SD_CD_list.append([SD_CD_a, SD_CD_c])

        self.evals_AB_list.append([evals_AB_a, evals_AB_c])
        self.evals_CD_list.append([evals_CD_a, evals_CD_c])

        self.evects_AB_list.append([evects_AB_a, evects_AB_c])
        self.evects_CD_list.append([evects_CD_a, evects_CD_c])

        return self

    def preprocess(self, meshB_list, meshD_list, n_ev=(50,50), elims=(-np.log(3),np.log(3)),
                   n_descr=100, scale=1.5, remove_area=False, remove_conformal=False,
                   SD_type='spectral', mapping=None, subsample_step=1, trim_values=False,
                   trim_scale=2, verbose=False):
        """
        Precompute descriptors for the matching pipeline. Collections are supposed to be aligned

        Parameters
        -----------------------------
        meshB_list  : (m,) list of TriMesh or mesh path to use for collection 1
        meshD_list  : (m,) list of TriMesh or mesh path to use for collection 1
        n_ev        : (n_ev1, n_ev2) tuple - with the number of Laplacian eigenvalues to consider
                      for each collection
        elims       : limits for energy values of DWKS descriptors
        n_descr     : int - number of descriptors to consider (ie number of energy values)
        scale       : float -sigma is expressed as ```scale * de/100``` where ```de``` is the
                      range of energy values
        remove_area : bool - if True, removes descriptor associated to the area shape difference
                      operator
        remove_confonformal : bool - if True, removes descriptor associated to the area shape
                              difference operator
        SD_type     : str - 'spectral' or 'semican', method to use to compute SDO
        mapping     : (m,) list of tuples ((n_B), (n_D)) with first element being the pointwise map
                      between meshA and meshB, and second element the pointwise map between meshC
                      and meshD (where meshB and meshD depend on the index of the list)
        subsample_step : rate at which to subsample descriptros
        trim_values : bool - whether to remove energy values too far away from maximal and minimal
                      eigenvalues for faster computation
        trim_scale  : new energy values will be set between
                      ```ev_min - trim_scale*sigma, ev_max + trim_scale*sigma```
                      with ev_min the minimum considered SDO eigenvalue, and ev_max the maximum one
        """

        # Compute the Laplacian spectrum
        assert len(meshB_list) == len(meshD_list), 'Use the same number of meshes'
        assert not remove_area or not remove_conformal, 'Use at least one type of descriptor'

        self.k1, self.k2 = n_ev
        self.mapping = mapping
        self.SD_type = SD_type
        self.remove_area = remove_area
        self.remove_conformal = remove_conformal

        # Compute laplacian on base shapes
        if verbose:
            print('\nComputing Laplacian spectrum')
        if self.meshA.eigenvalues is None or len(self.meshA.eigenvalues) < self.k1:
            self.meshA.process(max(self.k1, 200),verbose=verbose)
        if self.meshC.eigenvalues is None or len(self.meshC.eigenvalues) < self.k2:
            self.meshC.process(max(self.k2, 200),verbose=verbose)

        k_B = 0 if SD_type == 'semican' else 3*n_ev[0]
        k_D = 0 if SD_type == 'semican' else 3*n_ev[1]

        # Set energy values
        e_min, e_max = elims
        sigma = scale*(e_max-e_min)/100
        energy_list = np.linspace(e_min, e_max, n_descr)

        if verbose:
            print('\nProcessing Descriptors :')

        self.descr1 = np.empty((self.meshA.n_vertices,0))
        self.descr2 = np.empty((self.meshC.n_vertices,0))

        for meshind, (meshB_path, meshD_path) in enumerate(zip(meshB_list,meshD_list)):

            # Obtain meshes
            meshB = self.get_processed_mesh(meshB_path, k_B)
            meshD = self.get_processed_mesh(meshD_path, k_D)

            if verbose:
                print(f'\n\tMesh {meshind+1}/{len(meshB_list)} : {meshB.meshname}, {meshD.meshname}')

            current_maps = None if self.mapping is None else self.mapping[meshind]

            if verbose:
                print('\tComputing SD Operators')
            self.compute_SD(meshB, meshD, k_A=self.k1, k_C=self.k2, maps=current_maps,
                            SD_type=SD_type, verbose=verbose)

            if verbose:
                print('\tComputing descriptors')

            evals_AB = self.evals_AB_list[meshind]
            evects_AB = self.evects_AB_list[meshind]

            evals_CD = self.evals_CD_list[meshind]
            evects_CD = self.evects_CD_list[meshind]

            # Compute DWKS using the SD spectrum
            if not remove_area:
                descr1_a = sg.WKS(evals_AB[0], self.decode(evects_AB[0], mesh_ind=1), energy_list,
                                  sigma, scaled=False)
                descr2_a = sg.WKS(evals_CD[0], self.decode(evects_CD[0], mesh_ind=2), energy_list,
                                  sigma, scaled=False)

            if not remove_conformal:
                descr1_c = sg.WKS(evals_AB[1], self.decode(evects_AB[1], mesh_ind=1), energy_list,
                                  sigma, scaled=False)
                descr2_c = sg.WKS(evals_CD[1], self.decode(evects_CD[1], mesh_ind=2), energy_list,
                                  sigma, scaled=False)

            # Potentially remove near-0 values (happens when away from log-eigenvalues)
            if trim_values:
                new_einds_a = np.zeros_like(energy_list)
                if not remove_area:
                    new_emin_a = np.log(max(1e-3, min(np.min(evals_AB[0]), np.min(evals_CD[0])))) - trim_scale*sigma
                    new_emax_a = np.log(max(np.max(evals_AB[0]), np.max(evals_CD[0]))) + trim_scale*sigma
                    new_einds_a = (new_emin_a < energy_list) & (energy_list < new_emax_a)
                    descr1_a = descr1_a[:,new_einds_a]
                    descr2_a = descr2_a[:,new_einds_a]

                new_einds_c = np.zeros_like(energy_list)
                if not remove_conformal:
                    new_emin_c = np.log(max(1e-3, min(np.min(evals_AB[1]), np.min(evals_CD[1])))) - trim_scale*sigma
                    new_emax_c = np.log(max(np.max(evals_AB[1]), np.max(evals_CD[1]))) + trim_scale*sigma
                    new_einds_c = (new_emin_c < energy_list) & (energy_list < new_emax_c)
                    descr1_c = descr1_c[:,new_einds_c]
                    descr2_c = descr2_c[:,new_einds_c]

                if verbose:
                    print(f'\tRemoving {100*((1-new_einds_a).sum())/n_descr}% of area descriptors, '
                          f'{100*(1-new_einds_c).sum()/n_descr}% of conformal descriptors')

            if verbose:
                print('\tNormalizing descriptors')
            # Normalize descriptors
            if not remove_area:
                no1_a = np.sqrt(self.meshA.l2_sqnorm(descr1_a)).sum()
                no2_a = np.sqrt(self.meshC.l2_sqnorm(descr2_a)).sum()
                self.descr1 = np.hstack([self.descr1,descr1_a/no1_a])
                self.descr2 = np.hstack([self.descr2,descr2_a/no2_a])

            if not remove_conformal:
                no1_c = np.sqrt(self.meshA.l2_sqnorm(descr1_c)).sum()
                no2_c = np.sqrt(self.meshC.l2_sqnorm(descr2_c)).sum()
                self.descr1 = np.hstack([self.descr1, descr1_c/no1_c])
                self.descr2 = np.hstack([self.descr2, descr2_c/no2_c])

        if subsample_step > 1:
            self.descr1 = self.descr1[:,np.arange(0, self.descr1.shape[1], subsample_step)]
            self.descr2 = self.descr2[:,np.arange(0, self.descr2.shape[1], subsample_step)]

        if verbose:
            use_c = not remove_conformal
            use_a = not remove_area
            print(f'\n\t{self.descr1.shape[1]} out of {(2+int(use_c)+int(use_a))*n_descr*len(meshB_list)} possible descriptors kept')
            print('\tDone')

        return self

    def fit(self, descr_mu=1e1, lap_mu=1e-2, descr_comm_mu=1e-1, SD_comm_mu=1, orient_mu=0,
            optinit='random', verbose=False):
        """
        Solves the functional mapping problem and saves the computed Functional Map.

        Parameters
        -------------------------------
        descr_mu  : the scaling of the descriptor loss
        lap_mu    : the scaling of the laplacian commutativity loss
        comm_mu   : the scaling of the descriptor commutativity loss
        SD_comm_mu: the scaling of the SDO commutativity loss
        orient_mu : the scaling of the descriptor orientation
        opt_type  : 'scipy|cvxpy' which library to use for opt
        """
        self.change_FM_type('classic')
        if optinit not in ['random','identity', 'zeros']:
            raise ValueError(f"optinit arg should be 'random', 'identity' or 'zeros', not {optinit}")

        # Project the descriptors on the LB basis
        descr1_red = self.project(self.descr1, mesh_ind=1)  # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2, mesh_ind=2)  # (n_ev2, n_descr)

        # Compute multiplicative operators associated to each descriptor
        list_descr = []
        if descr_comm_mu > 0:
            if verbose:
                print('Computing commutativity operators')
            list_descr = self.compute_new_descr()  # (n_descr, ((k1,k1), (k2,k2)) )

        # Compute orientation operators associated to each descriptor
        orient_op = []
        if orient_mu > 0:
            if verbose:
                print('Computing orientation operators')
            orient_op = self.compute_orientation_op()  # (n_descr,)

        # List the SD operator we expect the FM to commute with
        SD_list = []
        if not self.remove_area:
            SD_list += [(x[0],y[0]) for x, y in zip(self.SD_AB_list, self.SD_CD_list)]
        if not self.remove_conformal:
            SD_list += [(x[1],y[1]) for x, y in zip(self.SD_AB_list, self.SD_CD_list)]

        # Compute the squared differences between eigenvalues for LB commutativity
        ev_sqdiff = np.square(self.meshA.eigenvalues[None,:self.k1] - self.meshC.eigenvalues[:self.k2,None])  # (n_ev2,n_ev1)
        ev_sqdiff /= np.linalg.norm(ev_sqdiff)**2

        # rescale orientation term
        if orient_mu > 0:
            args_native = (np.eye(self.k2,self.k1),
                           descr_mu, lap_mu, descr_comm_mu, 0, 0,
                           descr1_red, descr2_red,
                           list_descr, orient_op, ev_sqdiff, SD_list)
            eval_native = SD_opt.SD_energy_func_std(*args_native)
            eval_orient = opt_func.oplist_commutation(np.eye(self.k2,self.k1), orient_op)
            orient_mu *= eval_native / eval_orient
            if verbose:
                print(f'\tScaling orientation preservation weight by {eval_native / eval_orient:.1e}')

        # Arguments for the optimization problem
        args = (descr_mu, lap_mu, descr_comm_mu, SD_comm_mu, orient_mu,
                descr1_red, descr2_red,list_descr, orient_op, ev_sqdiff, SD_list)

        # Initialization
        x0 = self.get_x0(optinit=optinit)

        if verbose:
            print(f'\nOptimization :\n'
                  f'\t{self.k1} Ev on source - {self.k2} Ev on Target\n'
                  f'\tUsing {len(self.SD_AB_list)} meshes and {len(SD_list)} SD operators computed using {self.SD_type} method\n'
                  f'\tUsing {self.descr1.shape[1]} Descriptors\n'
                  f'\tHyperparameters :\n'
                  f'\t\tDescriptors preservation :{descr_mu:.1e}\n'
                  f'\t\tDescriptors commutativity :{descr_comm_mu:.1e}\n'
                  f'\t\tLaplacian commutativity :{lap_mu:.1e}\n'
                  f'\t\tShape Difference Commutativity :{SD_comm_mu:.1e}\n'
                  f'\t\tOrientation preservation :{orient_mu:.1e}\n'
                  )

        start_time = time.time()
        res = fmin_l_bfgs_b(SD_opt.SD_energy_func_std, x0.reshape(-1), fprime=SD_opt.grad_energy_std, args=args)
        opt_time = time.time() - start_time
        self.FM = res[0].reshape((self.k2, self.k1))

        if verbose:
            print("\tTask : {task}, funcall : {funcalls}, nit : {nit}, warnflag : {warnflag}".format(**res[2]))
            print(f'\tDone in {opt_time:.2f} seconds')

        return self

    def get_x0(self, optinit="zeros"):
        """
        Returns the initial functional map for optimization.

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.
                  In any case, the first column of the functional map is computed by hand
                  and not modified during optimization

        Output
        ------------------------
        x0 : corresponding initial vector
        """
        if optinit == 'random':
            x0 = np.random.random((self.k2, self.k1))
        elif optinit == 'identity':
            x0 = np.eye(self.k2, self.k1)
        else:
            x0 = np.zeros((self.k2, self.k1))

        # Sets the equivalence between the constant functions
        ev_sign = np.sign(self.meshA.eigenvectors[0,0]*self.meshC.eigenvectors[0,0])
        area_ratio = np.sqrt(self.meshC.area/self.meshA.area)

        x0[:,0] = np.zeros(self.k2)
        x0[0,0] = ev_sign * area_ratio

        return x0

    def icp_refine(self, nit=5, tol=None, overwrite=True,verbose=False):
        """
        Refines the functional map using ICP and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations to do
        overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
                    will be the icp refined FM
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        self._FM_icp = icp.icp_refine(self.meshA.eigenvectors[:,:self.k1], self.meshC.eigenvectors[:,:self.k2], self.FM, nit=nit, tol=tol, verbose=verbose)
        if overwrite:
            self.change_FM_type('icp')
        return self

    def zoomout_refine(self, nit=10,step=1, subsample=None, use_ANN=False, overwrite=True, verbose=False):
        """
        Refines the functional map using ZoomOut and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations to do
        step      : increase in dimension at each Zoomout Iteration
        subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
        use_ANN   : bool - If True, use approximate nearest neighbor
        overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                    will be the zoomout refined FM (larger than the other 2)
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        if subsample is None or subsample == 0:
            sub = None
        else:
            sub1 = self.meshA.extract_fps(subsample)
            sub2 = self.meshC.extract_fps(subsample)
            sub = (sub1,sub2)

        self._FM_zo = zoomout.mesh_zoomout_refine(self.meshA, self.meshC, self.FM, nit,
                                                  step=step, subsample=sub, use_ANN=use_ANN, verbose=verbose)
        if overwrite:
            self.FM_type = 'zoomout'
        return self

    def compute_new_descr(self):
        """
        Compute the multiplication operators associated with the descriptors

        Output
        ---------------------------
        operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
        """
        pinv1 = self.meshA.eigenvectors[:,:self.k1].T @ self.meshA.A  # (k1,n)
        pinv2 = self.meshC.eigenvectors[:,:self.k2].T @ self.meshC.A  # (k2,n)

        list_descr = [
                      (pinv1@(self.descr1[:,i,None]*self.meshA.eigenvectors[:,:self.k1]),
                       pinv2@(self.descr2[:,i,None]*self.meshC.eigenvectors[:,:self.k2])
                       )
                      for i in range(self.descr1.shape[1])
                      ]

        return list_descr

    def compute_orientation_op(self, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ---------------------------------
        reversing : whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Output
        ---------------------------------
        list_op : (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
                  (k2,k2) which represent operators supposed to commute.
        """
        n_descr = self.descr1.shape[1]

        # Precompute the inverse of the eigenvectors matrix
        pinv1 = self.meshA.eigenvectors[:,:self.k1].T @ self.meshA.A  # (k1,n)
        pinv2 = self.meshC.eigenvectors[:,:self.k2].T @ self.meshC.A  # (k2,n)

        # Compute the gradient of each descriptor
        grads1 = [self.meshA.gradient(self.descr1[:,i], normalize=normalize) for i in range(n_descr)]
        grads2 = [self.meshC.gradient(self.descr2[:,i], normalize=normalize) for i in range(n_descr)]

        # Compute the operators in reduced basis
        can_op1 = [pinv1 @ self.meshA.orientation_op(gradf) @ self.meshA.eigenvectors[:, :self.k1]
                   for gradf in grads1]

        if reversing:
            can_op2 = [- pinv2 @ self.meshC.orientation_op(gradf) @ self.meshC.eigenvectors[:, :self.k2]
                       for gradf in grads2]
        else:
            can_op2 = [pinv2 @ self.meshC.orientation_op(gradf) @ self.meshC.eigenvectors[:, :self.k2]
                       for gradf in grads2]

        list_op = list(zip(can_op1,can_op2))

        return list_op

    def get_processed_mesh(self,mesh_ref,K):
        if type(mesh_ref) is str or type(mesh_ref) is np.str_:
            processed_mesh = TriMesh(path=mesh_ref).process(K, verbose=False)
        else:
            processed_mesh = copy.deepcopy(mesh_ref)
            if processed_mesh.eigenvalues is None or len(processed_mesh.eigenvalues) < K:
                processed_mesh.process(K, verbose=False)

        return processed_mesh

    def project(self, func, k=None, mesh_ind=1):
        """
        Projects a function on the LB basis

        Parameters
        -----------------------
        func    : array - (n1|n2,p) evaluation of the function
        mesh_in : int  1 | 2 index of the mesh on which to encode

        Output
        -----------------------
        encoded_func : (n1|n2,p) array of decoded f
        """
        if k is None:
            k = self.k1 if mesh_ind == 1 else self.k2

        if mesh_ind == 1:
            return self.meshA.project(func,k=k)
        elif mesh_ind == 2:
            return self.meshC.project(func,k=k)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def decode(self, encoded_func, mesh_ind=2):
        """
        Decode a function from the LB basis

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        mesh_ind     : int  1 | 2 index of the mesh on which to decode

        Output
        -----------------------
        func : (n1|n2,p) array of decoded f
        """
        if mesh_ind == 1:
            return self.meshA.decode(encoded_func)
        elif mesh_ind == 2:
            return self.meshC.decode(encoded_func)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def transport(self, encoded_func, reverse=False):
        """
        transport a function from LB basis 1 to LB basis 2.
        If reverse is True, then the functions are transposed the other way
        using the transpose of the functional map matrix

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        reverse      : bool If true, transpose from 2 to 1 using the transpose of the FM

        Output
        -----------------------
        transp_func : (n2|n1,p) array of new encoding of the functions
        """
        if not self.preprocessed:
            raise ValueError("The Functional map must be fit before transporting a function")

        if not reverse:
            return self.FM @ encoded_func
        else:
            return self.FM.T @ encoded_func

    def transfer(self, func, reverse=False):
        """
        Transfer a function from mesh1 to mesh2.
        If 'reverse' is set to true, then the transfer goes
        the other way using the transpose of the functional
        map as approximate inverser transfer.

        Parameters
        ----------------------
        func : (n1|n2,p) evaluation of the functons

        Output
        -----------------------
        transp_func : (n2|n1,p) transfered function

        """
        if not reverse:
            return self.decode(self.transport(self.project(func)))

        else:
            encoding = self.project(func, mesh_ind=2)
            return self.decode(self.transport(encoding,reverse=True),
                               mesh_ind=1
                               )
