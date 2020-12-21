import os
import json
import numpy as np
import quantecon as qe
from util import sojourn_time, fill_empty_diagonals, chi2, homogeneity, lag_categorical
from ergodic import fmpt, steady_state


__ALL__ = ["Markov", "SpatialMarkov"]


class Markov(object):
    def __init__(self, class_ids, classes=None, mask=None, fill_empty_classes=False):
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.unique(class_ids)

        k = len(self.classes)
        class_ids = np.array(class_ids) if not isinstance(class_ids, np.ndarray) else class_ids
        transitions = np.zeros((k, k))
        b = 1 if mask is None else mask.ravel()
        for ii in range(len(class_ids) - 1):
            np.add.at(transitions, (class_ids[ii].ravel(), class_ids[ii + 1].ravel()), b)
        self.transitions = transitions
        self.p = transitions / np.clip(transitions.sum((-1), keepdims=True), a_min=1, a_max=None)
        if fill_empty_classes:
            self.p = fill_empty_diagonals(self.p)

        p_tmp = self.p
        p_tmp = fill_empty_diagonals(p_tmp)
        markovchain = qe.MarkovChain(p_tmp)
        self.num_cclasses = markovchain.num_communication_classes
        self.num_rclasses = markovchain.num_recurrent_classes

        self.cclasses_indices = markovchain.communication_classes_indices
        self.rclasses_indices = markovchain.recurrent_classes_indices
        transient = set(list(map(tuple, self.cclasses_indices))).difference(
            set(list(map(tuple, self.rclasses_indices)))
        )
        self.num_tclasses = len(transient)
        if len(transient):
            self.tclasses_indices = [np.asarray(i) for i in transient]
        else:
            self.tclasses_indices = None
        self.astates_indices = list(np.argwhere(np.diag(p_tmp) == 1))
        self.num_astates = len(self.astates_indices)

    @property
    def fmpt(self):
        if not hasattr(self, "_fmpt"):
            self._fmpt = fmpt(self.p, fill_empty_classes=True)
        return self._fmpt

    @property
    def steady_state(self):
        if not hasattr(self, "_steady_state"):
            self._steady_state = steady_state(self.p, fill_empty_classes=True)
        return self._steady_state

    @property
    def sojourn_time(self):
        if not hasattr(self, "_st"):
            self._st = sojourn_time(self.p)
        return self._st


class SpatialMarkov(object):
    def __init__(self, class_ids, classes=None, kernel_sz=5, sigma=None, fill_empty_classes=False):
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.unique(class_ids)

        self.kernel_sz = kernel_sz
        self.sigma = sigma
        self.fill_empty_classes = fill_empty_classes

        self.k = len(self.classes)
        self.m = self.k
        self.class_ids = np.array(class_ids) if not isinstance(class_ids, np.ndarray) else class_ids

        classic = Markov(class_ids, classes=self.classes, fill_empty_classes=self.fill_empty_classes)
        self.transitions = classic.transitions
        self.p = classic.p

        self.T, self.P = self._calc()

    def _calc(self):
        self.lclass_ids = lag_categorical(
            self.class_ids, classes=self.classes, kernel_sz=self.kernel_sz, sigma=self.sigma, T=True
        )

        T = np.zeros((self.m, self.k, self.k))
        for ii in range(len(self.class_ids) - 1):
            np.add.at(
                T, (self.lclass_ids[ii].ravel(), self.class_ids[ii].ravel(), self.class_ids[ii + 1].ravel()), 1,
            )

        P = T.copy()
        P = P / np.clip(P.sum((-1), keepdims=True), a_min=1, a_max=None)

        if self.fill_empty_classes:
            P = fill_empty_diagonals(P)

        return T, P

    @property
    def s(self):
        if not hasattr(self, "_s"):
            self._s = steady_state(self.p)
        return self._s

    @property
    def S(self):
        if not hasattr(self, "_S"):
            _S = []
            for i, p in enumerate(self.P):
                _S.append(steady_state(p))
            # if np.array(_S).dtype is np.dtype('O'):
            self._S = np.asarray(_S)
        return self._S

    @property
    def f(self):
        if not hasattr(self, "_f"):
            self._f = fmpt(self.p)
        return self._f

    @property
    def F(self):
        if not hasattr(self, "_F"):
            F = np.zeros_like(self.P)
            for i, p in enumerate(self.P):
                F[i] = fmpt(np.asarray(p))
            self._F = np.asarray(F)
        return self._F

    # bickenbach and bode tests
    @property
    def ht(self):
        if not hasattr(self, "_ht"):
            self._ht = homogeneity(self.T)
        return self._ht

    @property
    def Q(self):
        if not hasattr(self, "_Q"):
            self._Q = self.ht.Q
        return self._Q

    @property
    def Q_p_value(self):
        self._Q_p_value = self.ht.Q_p_value
        return self._Q_p_value

    @property
    def LR(self):
        self._LR = self.ht.LR
        return self._LR

    @property
    def LR_p_value(self):
        self._LR_p_value = self.ht.LR_p_value
        return self._LR_p_value

    @property
    def dof_hom(self):
        self._dof_hom = self.ht.dof
        return self._dof_hom

    # shtests
    @property
    def shtest(self):
        if not hasattr(self, "_shtest"):
            self._shtest = self._mn_test()
        return self._shtest

    @property
    def chi2(self):
        if not hasattr(self, "_chi2"):
            self._chi2 = self._chi2_test()
        return self._chi2

    @property
    def x2(self):
        if not hasattr(self, "_x2"):
            self._x2 = sum([c[0] for c in self.chi2])
        return self._x2

    @property
    def x2_pvalue(self):
        if not hasattr(self, "_x2_pvalue"):
            self._x2_pvalue = 1 - stats.chi2.cdf(self.x2, self.x2_dof)
        return self._x2_pvalue

    @property
    def x2_dof(self):
        if not hasattr(self, "_x2_dof"):
            k = self.k
            self._x2_dof = k * (k - 1) * (k - 1)
        return self._x2_dof

    def _mn_test(self):
        """
        helper to calculate tests of differences between steady state
        distributions from the conditional and overall distributions.
        """
        n0, n1, n2 = self.T.shape
        rn = list(range(n0))
        mat = [self._ssmnp_test(self.s, self.S[i], self.T[i].sum()) for i in rn]
        return mat

    def _ssmnp_test(self, p1, p2, nt):
        """
        Steady state multinomial probability difference test.
        Arguments
        ---------
        p1       :  array
                    (k, ), first steady state probability distribution.
        p1       :  array
                    (k, ), second steady state probability distribution.
        nt       :  int
                    number of transitions to base the test on.
        Returns
        -------
        tuple
                   (3 elements)
                   (chi2 value, pvalue, degrees of freedom)
        """

        o = nt * p2
        e = nt * p1
        d = np.multiply((o - e), (o - e))
        d = d / e
        chi2 = d.sum()
        pvalue = 1 - stats.chi2.cdf(chi2, self.k - 1)
        return (chi2, pvalue, self.k - 1)

    def _chi2_test(self):
        """
        helper to calculate tests of differences between the conditional
        transition matrices and the overall transitions matrix.
        """
        n0, n1, n2 = self.T.shape
        rn = list(range(n0))
        mat = [chi2(self.T[i], self.transitions) for i in rn]
        return mat


if __name__ == "__main__":
    prefix = "./data"

    with open(f"{prefix}/train.json", "r") as fp:
        name_lst = json.load(fp)

    classes = np.arange(2)

    from suitability import get_suitability, get_suitability_spatial

    for slice_i in name_lst["lesions"][:1]:
        lesion_seq = np.stack([np.load(x) for x in slice_i])
        # mm = Markov(lesion_seq[:2], classes=np.arange(2))
        mm = SpatialMarkov(lesion_seq[:2], classes=classes, kernel_sz=5)
        lsuit, suit = get_suitability_spatial(mm, lesion_seq[:2], classes=classes, kernel_sz=5)
