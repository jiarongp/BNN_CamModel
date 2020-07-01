import numpy as np


class EmpiricalBayesIsotropicPrior(object):
    """
    Bayesian linear regression where the hyper-parameters alpha and beta are estimated from the training data with Expectation Maximization (*Empirical Bayes*/*Type-II Maximum Lkelihood*)
    In this case the weight prior is an isotropic Gaussian distribution. The hyper-parameter alpha controls the variance of all weights.

    We assume that
    * The posterior for alpha and beta are sharply peaked around the optimal values \hat{alpha} and \hat{beta},
    * the prior for alpha and beta is relatively flat.
    """
    def __init__(self):
        self._alpha = None
        self._beta = None

        self._m_N = None
        self._S_N = None

    def _posterior(self, Phi, t, return_inverse=False):
        """Computes mean and covariance matrix of the posterior distribution."""

        # Specialized formula for isotropic Gaussian as prior
        # The first part is a symmetric matrix, the second part is also a symmetric matrix
        S_N_inv = self._alpha * np.eye(Phi.shape[1]) + self._beta * Phi.T.dot(Phi)
        # The inverse of a symmetric matrix must also be symmetric. But np.linalg.inv(S_N_inv) does not necessarily preserve symmetry.
        # As a remedy, compute the Cholesky decomposition, i.e., A = L * L.H, where L is lower-triangular and .H is the conjugate transpose operator.
        # This code does not check that S_N_inv is in fact Hermitian.
        L = np.linalg.cholesky(S_N_inv)
        # Compute inverse of L
        L_inv = np.linalg.inv(L)
        # Put together inverse of S_N_inv
        S_N = np.dot(L_inv.T, L_inv)

        m_N = self._beta * S_N.dot(Phi.T).dot(t)

        if return_inverse:
            return m_N, S_N, S_N_inv
        else:
            return m_N, S_N

    def fit(self, Phi, t, alpha_init=None, beta_init=None, max_iterations=100, verbose=True):
        # N is the number of samples, M the number of dimensions
        N, M = Phi.shape

        # Initialize parameters (as in scikit learn's Bayesian Ridge)
        eps = np.finfo(np.float64).eps
        self._alpha = 1. if alpha_init is None else alpha_init
        self._beta = 1 / (np.var(t) + eps) if beta_init is None else beta_init

        log_likelihood = self.log_marginal_likelihood(Phi, t)

        for i in range(max_iterations):
            previous_log_likelihood = log_likelihood

            # E-step: Compute the posterior of w given the current setting of the parameters alpha and beta
            self._m_N, self._S_N = self._posterior(Phi, t)

            # M-step: Maximize expected complete-data log likelihood w.r.t. alpha and beta
            self._alpha = M / (self._m_N.T @ self._m_N + np.trace(self._S_N))
            self._beta = N / (np.sum(np.square(t - (Phi @ self._m_N))) + np.trace(Phi.T @ Phi @ self._S_N))

            log_likelihood = self.log_marginal_likelihood(Phi, t)
            if verbose:
                iteration_fmt_str = ":0{}d".format(int(np.floor(np.log10(max_iterations - 1))) + 1)
                fmt_str = "[{" + iteration_fmt_str + "}] Log-likelihood = {:5.4f}, alpha = {:5.4f}, beta = {:5.4f}"
                print(fmt_str.format(i, log_likelihood, self._alpha, self._beta))

            assert log_likelihood >= previous_log_likelihood, "Likelihood is guaranteed to increase (or stagnate). Try casting your input to higher precision, i.e., `np.astype(Phi, np.float64)`."
            if log_likelihood <= previous_log_likelihood + 1e-3:
                break

        # Assert the covariance of weight posterior is symmetric
        assert np.allclose(self._S_N, self._S_N.T, atol=1e-6), "Expected weight posterior to have a symmetric covariance"

    def log_marginal_likelihood(self, Phi, t):
        """Computes the log of the marginal likelihood."""
        N, M = Phi.shape

        m_N, _, S_N_inv = self._posterior(Phi, t, return_inverse=True)

        E_D = self._beta * np.sum((t - Phi.dot(m_N)) ** 2)
        E_W = self._alpha * np.sum(m_N ** 2)

        score = M * np.log(self._alpha) + \
                N * np.log(self._beta) - \
                E_D - \
                E_W - \
                np.linalg.slogdet(S_N_inv)[1] - \
                N * np.log(2 * np.pi)

        return 0.5 * score

    def predict(self, Phi, return_std=False):
        """Computes mean and variances of the posterior predictive distribution."""
        y = Phi.dot(self._m_N)
        y_var = 1 / self._beta + np.sum(Phi.dot(self._S_N) * Phi, axis=1)

        if return_std:
            return y, np.sqrt(y_var)

        return y, y_var

    def log_likelihood(self, Phi, t):
        """
        Compute the log-likelihood for each example.
        Assuming independent data points, the likelihood is \prod\limits_{n=1}^N \mathcal{N}(t_n | y(x_n, w), \beta^{-1})
        :param Phi: design matrix, ndarray of shape [num_samples, num_feature_dims]
        :param t: target labels
        :return: log-likelihood for each example. Because we are in log-space already, sum over examples to compute the log-likelihood for a set of examples.
        """
        y = self.predict(Phi)[0]

        # Eq. (1.62)
        log_likelihoods = 0.5 * (np.log(self._beta) - np.log(2 * np.pi) - self._beta * np.square(y - t))

        # Eq. (1.62) for the whole data set
        N = len(t)
        eq = -self._beta / 2. * np.sum(np.square(y - t)) + N / 2. * np.log(self._beta) - N / 2. * np.log(2 * np.pi)
        assert np.isclose(eq, np.sum(log_likelihoods))

        # To compute the log-likelihood over multiple examples, sum over the individual items. The log has turned the product into a sum.
        return log_likelihoods


class EmpiricalBayesUnknownObservationNoise(EmpiricalBayesIsotropicPrior):
    """
    Bayesian linear regression with known precision of the weights but unknown observation noise
    * We assume that precision of the weights is known
    * We assume that the posterior distribution for beta is sharply peaked around some optimal value \hat{beta} and the prior for beta is relatively flat
    The optimal value \hat{beta} is obtained by maximizing the marginal likelihood.

    As an alternative, we can place a normal-gamma distribution over beta (see `known_hyperparameters.BayesianLinearRegressionUnknownObservationNoise`).
    """
    def __init__(self, alpha):
        super(EmpiricalBayesUnknownObservationNoise, self).__init__()
        self._alpha = alpha

    def fit(self, Phi, t, alpha_init=None, beta_init=None, max_iterations=100, verbose=True):
        if alpha_init is not None:
            raise ValueError("alpha is assumed to be known. We just kept the parameter to be consistent with the base class method.")

        # N is the number of samples, M the number of dimensions
        N, M = Phi.shape

        # Initialize parameters (as in scikit learn's Bayesian Ridge)
        eps = np.finfo(np.float64).eps
        self._beta = 1 / (np.var(t) + eps) if beta_init is None else beta_init

        log_likelihood = self.log_marginal_likelihood(Phi, t)

        for i in range(max_iterations):
            previous_log_likelihood = log_likelihood

            # E-step: Compute the posterior of w given the current setting of the parameters alpha and beta
            self._m_N, self._S_N = self._posterior(Phi, t)

            # M-step: Maximize expected complete-data log likelihood w.r.t. beta
            self._beta = N / (np.sum(np.square(t - (Phi @ self._m_N))) + np.trace(Phi.T @ Phi @ self._S_N))

            log_likelihood = self.log_marginal_likelihood(Phi, t)
            if verbose:
                iteration_fmt_str = ":0{}d".format(int(np.floor(np.log10(max_iterations - 1))) + 1)
                fmt_str = "[{" + iteration_fmt_str + "}] Log-likelihood = {:5.4f}, alpha = {:5.4f}, beta = {:5.4f}"
                print(fmt_str.format(i, log_likelihood, self._alpha, self._beta))

            assert log_likelihood >= previous_log_likelihood, "Likelihood is guaranteed to increase (or stagnate). Try casting your input to higher precision, i.e., `np.astype(Phi, np.float64)`."
            if log_likelihood <= previous_log_likelihood + 1e-3:
                break

        # Assert the covariance of weight posterior is symmetric
        assert np.allclose(self._S_N, self._S_N.T, atol=1e-6), "Expected weight posterior to have a symmetric covariance"
