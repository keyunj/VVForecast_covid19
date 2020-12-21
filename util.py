import numpy as np
import quantecon as qe
from scipy import stats

__all__ = [
    "sojourn_time",
    "chi2",
    "homogeneity",
    "fill_empty_diagonals",
    "lag_categorical",
]


def lag_categorical(y, classes=None, kernel_sz=9, sigma=None, T=False):
    if T:
        return np.stack([lag_categorical(x, classes=classes, kernel_sz=kernel_sz, sigma=sigma) for x in y])
    # neighborhood
    ndims = len(y.shape)
    if isinstance(kernel_sz, int):
        kernel_sz = np.array([kernel_sz,] * ndims)
    elif len(kernel_sz) > 1:
        kernel_sz = np.array(kernel_sz)
        assert len(kernel_sz) == ndims
    else:
        raise ValueError("Wrong type of kernel size")
    if sigma is None:
        sigma = kernel_sz / 3
    # offset and weightsg
    offset = [np.arange(-(x // 2), x - (x // 2)) for x in kernel_sz]
    gaussian = [np.exp(-(o ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s) for o, s in zip(offset, sigma)]
    offset = np.stack(np.meshgrid(*offset, indexing="ij")).reshape(ndims, -1)[:, :, None]
    gaussian = np.stack(np.meshgrid(*gaussian, indexing="ij")).prod((0)).reshape(1, -1)[:, :, None]
    #
    if classes is None:
        classes = np.unique(y)
    k = len(classes)
    indices = [np.arange(x) for x in y.shape]
    indices = np.stack(np.meshgrid(*indices, indexing="ij")).reshape(ndims, -1)[:, None]
    idx_offset = indices + offset
    for ii, upper in enumerate(y.shape):
        idx_offset[ii] = np.maximum(idx_offset[ii], 0)
        idx_offset[ii] = np.minimum(idx_offset[ii], upper - 1)

    lag = np.zeros((k, *(y.shape)))
    cnt = np.zeros_like(lag)
    cls_idx = y[tuple(idx_offset)].ravel()
    shape_idx = [x.repeat(offset.shape[1], 0).ravel() for x in indices]
    weights = gaussian.repeat(indices.shape[2], 2).ravel()
    np.add.at(lag, (cls_idx, *shape_idx), weights)
    np.add.at(cnt, (cls_idx, *shape_idx), 1)
    lag = lag / np.clip(cnt, a_min=1, a_max=None)

    return classes[tuple(np.argmax(lag, axis=0)[None,])]


def sojourn_time(p, summary=False):
    """
    Calculate sojourn time based on a given transition probability matrix.
    Parameters
    ----------
    p        : array
               (k, k), a Markov transition probability matrix.
    summary  : bool
               If True and the Markov Chain has absorbing states whose
               sojourn time is infinitely large, print out the information
               about the absorbing states. Default is True.
    Returns
    -------
             : array
               (k, ), sojourn times. Each element is the expected time a Markov
               chain spends in each state before leaving that state.
    Notes
    -----
    Refer to :cite:`Ibe2009` for more details on sojourn times for Markov
    chains.
    Examples
    --------
    >>> from giddy.markov import sojourn_time
    >>> import numpy as np
    >>> p = np.array([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])
    >>> sojourn_time(p)
    array([2., 1., 2.])
    Non-ergodic Markov Chains with rows full of 0
    >>> p = np.array([[.5, .25, .25], [.5, 0, .5],[ 0, 0, 0]])
    >>> sojourn_time(p)
    Sojourn times are infinite for absorbing states! In this Markov Chain, states [2] are absorbing states.
    array([ 2.,  1., inf])
    """

    p = np.asarray(p)
    if (p.sum(axis=1) == 0).sum() > 0:
        p = fill_empty_diagonals(p)

    markovchain = qe.MarkovChain(p)
    pii = p.diagonal()

    if not (1 - pii).all():
        absorbing_states = np.where(pii == 1)[0]
        non_absorbing_states = np.where(pii != 1)[0]
        st = np.full(len(pii), np.inf)
        if summary:
            print(
                "Sojourn times are infinite for absorbing states! In this "
                "Markov Chain, states {} are absorbing states.".format(list(absorbing_states))
            )
        st[non_absorbing_states] = 1 / (1 - pii[non_absorbing_states])
    else:
        st = 1 / (1 - pii)
    return st


def chi2(T1, T2):
    """
    chi-squared test of difference between two transition matrices.
    Parameters
    ----------
    T1    : array
            (k, k), matrix of transitions (counts).
    T2    : array
            (k, k), matrix of transitions (counts) to use to form the
            probabilities under the null.
    Returns
    -------
          : tuple
            (3 elements).
            (chi2 value, pvalue, degrees of freedom).
    Examples
    --------
    >>> import libpysal
    >>> from giddy.markov import Spatial_Markov, chi2
    >>> f = libpysal.io.open(libpysal.examples.get_path("usjoin.csv"))
    >>> years = list(range(1929, 2010))
    >>> pci = np.array([f.by_col[str(y)] for y in years]).transpose()
    >>> rpci = pci/(pci.mean(axis=0))
    >>> w = libpysal.io.open(libpysal.examples.get_path("states48.gal")).read()
    >>> w.transform='r'
    >>> sm = Spatial_Markov(rpci, w, fixed=True)
    >>> T1 = sm.T[0]
    >>> T1
    array([[562.,  22.,   1.,   0.],
           [ 12., 201.,  22.,   0.],
           [  0.,  17.,  97.,   4.],
           [  0.,   0.,   3.,  19.]])
    >>> T2 = sm.transitions
    >>> T2
    array([[884.,  77.,   4.,   0.],
           [ 68., 794.,  87.,   3.],
           [  1.,  92., 815.,  51.],
           [  1.,   0.,  60., 903.]])
    >>> chi2(T1,T2)
    (23.39728441473295, 0.005363116704861337, 9)
    Notes
    -----
    Second matrix is used to form the probabilities under the null.
    Marginal sums from first matrix are distributed across these probabilities
    under the null. In other words the observed transitions are taken from T1
    while the expected transitions are formed as follows
    .. math::
            E_{i,j} = \sum_j T1_{i,j} * T2_{i,j}/\sum_j T2_{i,j}
    Degrees of freedom corrected for any rows in either T1 or T2 that have
    zero total transitions.
    """
    rs2 = T2.sum(axis=1)
    rs1 = T1.sum(axis=1)
    rs2nz = rs2 > 0
    rs1nz = rs1 > 0
    dof1 = sum(rs1nz)
    dof2 = sum(rs2nz)
    rs2 = rs2 + (rs2 == 0)
    dof = (dof1 - 1) * (dof2 - 1)
    p = np.diag(1 / rs2).dot(np.array(T2))
    E = np.diag(rs1).dot(np.array(p))
    num = T1 - E
    num = np.multiply(num, num)
    E = E + (E == 0)
    chi2 = num / E
    chi2 = chi2.sum()
    pvalue = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, pvalue, dof


def fill_empty_diagonals(p):
    """
    Assign 1 to diagonal elements which fall in rows full of 0s to ensure
    the transition probability matrix is a stochastic one. Currently
    implemented for two- and three-dimensional transition probability
    matrices.
    Parameters
    ----------
    p        : array
               (k, k), an ergodic/non-ergodic Markov transition probability
               matrix.
    Returns
    -------
    p_temp   : array
               Matrix without rows full of 0 transition probabilities.
               (stochastic matrix)
    Examples
    --------
    >>> import numpy as np
    >>> from giddy.util import fill_empty_diagonals
    >>> p2 = np.array([[.5, .5, 0], [.3, .7, 0], [0, 0, 0]])
    >>> fill_empty_diagonals(p2)
    array([[0.5, 0.5, 0. ],
           [0.3, 0.7, 0. ],
           [0. , 0. , 1. ]])
    >>> p3 = np.array([[[0.5, 0.5, 0. ], [0.3, 0.7, 0. ], [0. , 0. , 0. ]],
    ...  [[0. , 0. , 0. ], [0.3, 0.7, 0. ], [0. , 0. , 0. ]]])
    >>> p_new = fill_empty_diagonals(p3)
    >>> p_new[1]
    array([[1. , 0. , 0. ],
           [0.3, 0.7, 0. ],
           [0. , 0. , 1. ]])
    """

    if len(p.shape) == 3:
        return fill_empty_diagonal_3d(p)
    elif len(p.shape) == 2:
        return fill_empty_diagonal_2d(p)
    else:
        raise NotImplementedError("Filling empty diagonals is only implemented for 2/3d matrices.")


def fill_empty_diagonal_2d(p):
    """
    Assign 1 to diagonal elements which fall in rows full of 0s to ensure
    the transition probability matrix is a stochastic one.
    Parameters
    ----------
    p        : array
               (k, k), an ergodic/non-ergodic Markov transition probability
               matrix.
    Returns
    -------
    p_temp   : array
               Matrix without rows full of 0 transition probabilities.
               (stochastic matrix)
    """

    p_tmp = p.copy()
    p0 = p_tmp.sum((1)) == 0
    if p0.sum() > 0:
        row_zero_i = np.where(p0)
        for row in row_zero_i:
            p_tmp[row, row] = 1
    return p_tmp


def fill_empty_diagonal_3d(p):
    """
    Assign 1 to diagonal elements which fall in rows full of 0s to ensure
    the conditional transition probability matrices is are stochastic matrices.
    Staying probabilities are 1.
    Parameters
    ----------
    p        : array
               (m, k, k), m ergodic/non-ergodic Markov transition probability
               matrices.
    Returns
    -------
    p_temp   : array
               Matrices without rows full of 0 transition probabilities.
               (stochastic matrices)
    """

    p_tmp = p.copy()
    p0 = p_tmp.sum((2)) == 0
    if p0.sum() > 0:
        rows, cols = np.where(p0)
        row_zero_i = list(zip(rows, cols))
        for row in row_zero_i:
            i, j = row
            p_tmp[i, j, j] = 1
    return p_tmp


def homogeneity(transition_matrices, regime_names=[], class_names=[], title="Markov Homogeneity Test"):
    """
    Test for homogeneity of Markov transition probabilities across regimes.
    Parameters
    ----------
    transition_matrices : list
                          of transition matrices for regimes, all matrices must
                          have same size (r, c). r is the number of rows in the
                          transition matrix and c is the number of columns in
                          the transition matrix.
    regime_names        : sequence
                          Labels for the regimes.
    class_names         : sequence
                          Labels for the classes/states of the Markov chain.
    title               : string
                          name of test.
    Returns
    -------
                        : implicit
                          an instance of Homogeneity_Results.
    """

    return Homogeneity_Results(transition_matrices, regime_names=regime_names, class_names=class_names, title=title)


class Homogeneity_Results:
    """
    Wrapper class to present homogeneity results.
    Parameters
    ----------
    transition_matrices : list
                          of transition matrices for regimes, all matrices must
                          have same size (r, c). r is the number of rows in
                          the transition matrix and c is the number of columns
                          in the transition matrix.
    regime_names        : sequence
                          Labels for the regimes.
    class_names         : sequence
                          Labels for the classes/states of the Markov chain.
    title               : string
                          Title of the table.
    Attributes
    -----------
    Notes
    -----
    Degrees of freedom adjustment follow the approach in :cite:`Bickenbach2003`.
    Examples
    --------
    See Spatial_Markov above.
    """

    def __init__(self, transition_matrices, regime_names=[], class_names=[], title="Markov Homogeneity Test"):
        self._homogeneity(transition_matrices)
        self.regime_names = regime_names
        self.class_names = class_names
        self.title = title

    def _homogeneity(self, transition_matrices):
        # form null transition probability matrix
        M = np.array(transition_matrices)
        m, r, k = M.shape
        self.k = k
        B = np.zeros((r, m))
        T = M.sum(axis=0)
        self.t_total = T.sum()
        n_i = T.sum(axis=1)
        A_i = (T > 0).sum(axis=1)
        A_im = np.zeros((r, m))
        p_ij = np.dot(np.diag(1.0 / (n_i + (n_i == 0) * 1.0)), T)
        den = p_ij + 1.0 * (p_ij == 0)
        b_i = np.zeros_like(A_i)
        p_ijm = np.zeros_like(M)
        # get dimensions
        m, n_rows, n_cols = M.shape
        m = 0
        Q = 0.0
        LR = 0.0
        lr_table = np.zeros_like(M)
        q_table = np.zeros_like(M)

        for nijm in M:
            nim = nijm.sum(axis=1)
            B[:, m] = 1.0 * (nim > 0)
            b_i = b_i + 1.0 * (nim > 0)
            p_ijm[m] = np.dot(np.diag(1.0 / (nim + (nim == 0) * 1.0)), nijm)
            num = (p_ijm[m] - p_ij) ** 2
            ratio = num / den
            qijm = np.dot(np.diag(nim), ratio)
            q_table[m] = qijm
            Q = Q + qijm.sum()
            # only use nonzero pijm in lr test
            mask = (nijm > 0) * (p_ij > 0)
            A_im[:, m] = (nijm > 0).sum(axis=1)
            unmask = 1.0 * (mask == 0)
            ratio = (mask * p_ijm[m] + unmask) / (mask * p_ij + unmask)
            lr = nijm * np.log(ratio)
            LR = LR + lr.sum()
            lr_table[m] = 2 * lr
            m += 1
        # b_i is the number of regimes that have non-zero observations in row i
        # A_i is the number of non-zero elements in row i of the aggregated
        # transition matrix
        self.dof = int(((b_i - 1) * (A_i - 1)).sum())
        self.Q = Q
        self.Q_p_value = 1 - stats.chi2.cdf(self.Q, self.dof)
        self.LR = LR * 2.0
        self.LR_p_value = 1 - stats.chi2.cdf(self.LR, self.dof)
        self.A = A_i
        self.A_im = A_im
        self.B = B
        self.b_i = b_i
        self.LR_table = lr_table
        self.Q_table = q_table
        self.m = m
        self.p_h0 = p_ij
        self.p_h1 = p_ijm
