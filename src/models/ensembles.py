import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        estimator=DecisionTreeRegressor, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        estimator : Estimator
            The child estimator template used to create the collection of fitted sub-estimators
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.tree_parameters = trees_parameters
        self.estimator = estimator
        self.feature_samples = []
        self.train_samples = []
        self.trees = []

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        trace : bool
            If True return (self, history), history is a dictionary with history of fitting
        """

        fit_times = []
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        train_compos_pred = np.zeros(X.shape[0], dtype=np.float)
        test_compos_pred = 0.
        train_rmses = []
        test_rmses = []

        start = time.time()

        for i in range(self.n_estimators):
            self.feature_samples.append(np.random.choice(X.shape[1], self.feature_subsample_size, replace=False))
            self.train_samples.append(np.random.randint(0, X.shape[0], X.shape[0], dtype=np.int32))
            new_tree = self.estimator(max_depth=self.max_depth, **self.tree_parameters)
            new_tree.fit(X[:, self.feature_samples[i]][self.train_samples[i]],
                         y[self.train_samples[i]])
            self.trees.append(new_tree)

            train_compos_pred += (self.trees[i].predict(X[:, self.feature_samples[i]]))
            train_rmses.append(mean_squared_error(y, train_compos_pred / (i + 1), squared=False))

            if X_val is not None:
                test_compos_pred += (self.trees[i].predict(X_val[:, self.feature_samples[i]]))
                test_rmses.append(mean_squared_error(y_val, test_compos_pred / (i + 1), squared=False))

            fit_times.append(time.time() - start)

        if trace:
            return self, \
                   {
                       'times': fit_times,
                       'train_rmses': train_rmses,
                       'test_rmses': test_rmses
                   }

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        pred_compos = np.zeros(X.shape[0], dtype=np.float)
        for i in range(self.n_estimators):
            pred_compos += self.trees[i].predict(X[:, self.feature_samples[i]])
        return pred_compos / self.n_estimators


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        estimator=DecisionTreeRegressor, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        estimator : Estimator
            The child estimator template used to create the collection of fitted sub-estimators
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.estimator = estimator
        self.tree_parametrs = trees_parameters
        self.trees = []
        self.feature_samples = []
        self.weights = []
        self.train_samples = []

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        trace : bool
            If True return (self, history), history is a dictionary with history of fitting
        """

        fit_times = []
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        train_rmses = []
        test_rmses = []
        f = np.zeros(y.size, dtype=np.float)

        start = time.time()

        for i in range(self.n_estimators):
            self.feature_samples.append(np.random.choice(X.shape[1],
                                        self.feature_subsample_size, replace=False))
            train_sample = np.random.choice(X.shape[0], (X.shape[0] * 7) // 10, replace=False)
            new_tree = self.estimator(max_depth=self.max_depth, **self.tree_parametrs)
            if i == 0:
                self.trees.append(new_tree.fit(X[:, self.feature_samples[i]][train_sample], y[train_sample]))
                self.weights.append(1.0)
                f = self.trees[i].predict(X[:, self.feature_samples[i]])
            else:
                self.trees.append(new_tree.fit(X[:, self.feature_samples[i]][train_sample], 2 * (y - f)[train_sample]))
                y_pred = self.trees[i].predict(X[:, self.feature_samples[i]])
                alpha_t = minimize_scalar(lambda x: mean_squared_error(y,
                                                                       f + x * y_pred,
                                                                       squared=False)).x
                self.weights.append(self.learning_rate * alpha_t)
                f = f + self.weights[i] * y_pred

            train_rmses.append(mean_squared_error(y, f, squared=False))
            if X_val is not None:
                pred = np.zeros(X_val.shape[0])
                for j in range(i + 1):
                    pred += self.weights[j] * self.trees[j].predict(X_val[:, self.feature_samples[j]])
                test_rmses.append(mean_squared_error(y_val,
                                                     pred,
                                                     squared=False))
            fit_times.append(time.time() - start)

        if trace:
            return self, \
                   {
                       'times': fit_times,
                       'train_rmses': train_rmses,
                       'test_rmses': test_rmses
                   }

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            pred += self.weights[i] * self.trees[i].predict(X[:, self.feature_samples[i]])
        return pred
