import numpy as np


def kl_divergence_multivariate_gaussians(mu_a, mu_b, cov_a, cov_b):
    """
    Compute the KL divergence between two multivariate Gaussians, i.e. KL(P_a || P_b)

    Derivation: See https://stanford.edu/~jduchi/projects/general_notes.pdf
    :param mu_a: mean of distribution a
    :param mu_b: mean of distribution b
    :param cov_a: covariance of distribution a
    :param cov_b: covariance of distribution b
    :return: KL(P_a || P_b)
    """
    num_dims = len(mu_a)
    cov_b_inv = np.linalg.inv(cov_b)

    return 0.5 * (
        + np.linalg.slogdet(cov_b)[1] - np.linalg.slogdet(cov_a)[1]
        - num_dims
        + np.trace(cov_b_inv @ cov_a)
        + (mu_b - mu_a).T @ (cov_b_inv @ (mu_b - mu_a))
    )


def polynomial_basis_function(x, degree):
    return x ** degree


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones((len(x), 1)), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)
