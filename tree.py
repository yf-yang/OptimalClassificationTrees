from abc import ABCMeta

import numpy as np
import cvxpy as cvx
from scipy.stats import mode

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.externals import six
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

class OptimalClassificationTree(six.with_metaclass(ABCMeta, BaseEstimator), ClassifierMixin):

    def __init__(self, 
                 max_depth, 
                 min_samples_leaf=1,
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y, solver="GUROBI", eps=0.005, cp=0):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
        """

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            # if self.class_weight is not None:
            #     y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=np.int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                       return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            # if self.class_weight is not None:
            #     expanded_class_weight = compute_sample_weight(
            #         self.class_weight, y_original)

        else:
            raise Exception("No support for regression.")
        self.n_classes_ = self.n_classes_[0]
        y = _one_hot(y, self.n_classes_)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')

        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        # normalize to [0,1]
        self.scaler = MinMaxScaler()

        self.n_branch_nodes = (1<<self.max_depth) - 1
        self.n_leaf_nodes = 1 << self.max_depth

        # cvxpy variable wrapper
        v_a = cvx.Variable(shape=(self.n_branch_nodes, self.n_features_),
                           name='a')
        v_b = cvx.Variable(shape=self.n_branch_nodes, name='b')
        v_d = cvx.Variable(shape=self.n_branch_nodes, name='d', boolean=True)
        v_s = cvx.Variable(shape=(self.n_branch_nodes, self.n_features_), 
                           name='s', boolean=True)
        v_z = cvx.Variable(shape=(self.n_leaf_nodes, n_samples), name='z',
                           boolean=True)
        v_l = cvx.Variable(shape=self.n_leaf_nodes, name='l', boolean=True)
        v_Nt = cvx.Variable(shape=self.n_leaf_nodes, name='Nt', integer=True)
        v_Ntk = cvx.Variable(shape=(self.n_leaf_nodes, self.n_classes_),
                           name='Ntk', integer=True)
        v_Lt = cvx.Variable(shape=self.n_leaf_nodes, name='Lt')
        v_c = cvx.Variable(shape=(self.n_classes_, self.n_leaf_nodes),
                           name='c', boolean=True)

        X = self.scaler.fit_transform(X)

        children, parents = _get_parent(self.n_branch_nodes)

        (self.path, self.path_encoding, 
            left_ancestors, right_ancestors) = _gen_leaf_path(self.max_depth)

        self.path = self.path.flatten()

        # constraints
        constraints = [
            cvx.abs(v_a) <= v_s,
            cvx.max(v_s, axis=1) <= v_d,
            cvx.sum(v_s, axis=1) >= v_d,
            cvx.abs(v_b) <= v_d,
            # cvx.max(v_s, axis=1) >= v_d,
            # *[
            #     v_s[:,j] >= v_d for j in range(self.n_features_)
            # ],
            cvx.sum(v_s, axis=1) <= self.max_features_,
            cvx.max(v_z, axis=1) <= v_l,
            cvx.sum(v_z, axis=1) >= self.min_samples_leaf * v_l,
            v_d[children] <= v_d[parents],
            cvx.sum(v_z, axis=0) == 1,
            *[
                X * v_a[m,:] + eps <= v_b[m] + (2+eps) * (1-v_z[t])
                    for t, m in left_ancestors
            ],
            *[
                X * v_a[m,:] >= v_b[m] - 2 * (1-v_z[t])
                    for t, m in right_ancestors
            ],
            cvx.sum(v_z, axis=1) == v_Nt,
            v_Ntk == v_z * y,
            *[
                v_Lt >= v_Nt - v_Ntk[:,k] - n_samples * (1-v_c[k])
                    for k in range(self.n_classes_)
            ],
            *[
                v_Lt <= v_Nt - v_Ntk[:,k] + n_samples * v_c[k]
                    for k in range(self.n_classes_)
            ],
            cvx.sum(v_c, axis=0) == v_l,
            v_Lt >= 0,
        ]

        # baseline loss
        base_loss = y.size - mode(y, axis=None)[1][0]

        # objective
        objective = cvx.Minimize(cvx.sum(v_Lt) / base_loss
                    + cp * cvx.sum(v_s))

        # MIP solver
        if solver == "GUROBI":
            cvx_solver = cvx.GUROBI
        else:
            raise ValueError("No support for non GUROBI MIP solver.")

        if solver not in cvx.installed_solvers():
            raise ValueError("Installed solvers: %s, got %s"
                             % (cvx.installed_solvers(), solver))

        # solve the problem
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=solver)

        # save tree parameters
        self.a = v_a.value.T
        self.b = v_b.value
        self.depth = self.max_depth # TBD

        label, active_leaf = np.where(v_c.value)
        self.leaf_class = -np.ones(self.n_leaf_nodes, dtype=np.int)
        self.leaf_class[active_leaf] = label

        # self.prob = v_Ntk.value / v_Nt.value.reshape(-1,1)

        return self

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # TBD check fitted
        n_samples, n_features = X.shape

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        X = self.scaler.fit_transform(X)

        ge = X.dot(self.a) >= self.b

        # vectorization (space for time)
        import pdb; pdb.set_trace()
        path = np.tile(ge, self.n_leaf_nodes)[:,self.path.flatten()]
        eq = (path == self.path_encoding.flatten())
        leaf = np.where(eq.reshape(n_samples, -1, self.depth).all(axis=2))[1]

        assert leaf.shape == (n_samples,)

        y = self.leaf_class[leaf]

        assert np.all(y!=-1)

        return y

def _get_parent(n_branch_nodes):
    """ Generate mapping from every branch nodes but root to there parent

    Parameters
    ----------
    n_branch_nodes : int
        Number of branch nodes.

    Returns
    -------
    children : array-like, shape = [n_branch_nodes-1]
        List of child branch nodes.

    parents : array-like, shape = [n_branch_nodes-1]
        List of parent nodes of corresponding child nodes.
    """
    children = np.arange(1, n_branch_nodes)
    parents = (children-1) // 2
    return children, parents

def _gen_leaf_path(depth):
    """ Generate paths from tree root to each leaf.

    Parameters
    ----------
    depth : int
        The depth of the tree.

    Returns
    -------
    mask : array-like, shape = [2**depth, 2**depth-1]
        Each row is a path of a leaf node, each column index denotes whether
        the path go by that branch node. So exact ``depth`` elements in a row
        are True, others are False.

    route : array-like, shape = [2**depth, depth]
        Each row is the set of choices taken when proceed from root to a leaf 
        node. False denotes a left child, True denotes a right child.

    left : a list of tuple (leaf_idx, branch_idx)
        Mapping of leaf node to branch node, where the branch node is the leaf
        node's left ancestor.

    right : a list of tuple (leaf_idx, branch_idx)
        Mapping of leaf node to branch node, where the branch node is the leaf
        node's right ancestor.
    """

    mask = np.hstack(np.eye(1<<i, dtype=np.bool).repeat(1<<(depth-i), axis=0) 
                        for i in range(depth))
    route = np.column_stack(np.arange(1<<depth)//(1<<(depth-i-1))%2 
                                for i in range(depth)).astype(np.bool)

    leaf_idx, branch_idx = np.where(mask)
    left = list(zip(leaf_idx[~route.flatten()], branch_idx[~route.flatten()]))
    right = list(zip(leaf_idx[route.flatten()], branch_idx[route.flatten()]))

    return mask, route, left, right

def _one_hot(tensor, n_classes):
    tensor = np.squeeze(tensor)
    tensor_flat = tensor.flatten()
    n_sample = len(tensor_flat)
    encoded_flat = np.zeros((n_sample, n_classes), dtype = np.int)
    encoded_flat[np.arange(n_sample), tensor_flat] = 1
    encoded = encoded_flat.reshape(*tensor.shape, n_classes)
    return encoded