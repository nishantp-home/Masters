import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def gmm(X, K, max_iter=100, tol=1e-5, smoothing=1e-2):
    N, D = X.shape    
    M = np.zeros((K, D))    # Means
    R = np.zeros((N, K))    # Responsibility matrix (also represented with symbol gamma)
    C = np.zeros((K, D, D)) # Covariance matrix
    pi = np.ones(K) / K    # uniform

# Initialize M to random, initialize C to spherical with variance 1
    for k in range(K):
        M[k] = X[np.random.choice(N)]
        C[k] = np.diag(np.ones(D))


    lls = []
    weighted_pdfs = np.zeros((N, K))

    for i in range(max_iter):
        # Step 1: Determine assignments / responsibilities
        # This is the slow way
        # for k in range(K):
        #     for n in range(N):
        #         weighted_pdfs[n, k] = pi[k]*multivariate_normal.pdf(X[n], M[k], C[k])

        # for k in range(K):
        #     for n in range(N):
        #         R[n, k] = weighted_pdfs[n, k] / weighted_pdfs[n,:].sum()
        
        # step 1: Faster way with vectorization
        for k in range(K):
            weighted_pdfs[:, k] = pi[k]*multivariate_normal.pdf(X, M[k], C[k])
        R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

        # Step 2: Recalculate params
        for k in range(K):
            Nk = R[:, k].sum()
            pi[k] = Nk / N
            M[k] = R[:, k].dot(X) / Nk

            delta = X - M[k]
            Rdelta = np.expand_dims(R[:,k], -1) * delta  # multiplies R[:,k] by each col. of delta - N x D
            C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D)* smoothing   # D x D

        ll = np.log(weighted_pdfs.sum(axis=1)).sum()
        lls.append(ll)
        if i > 0:
            if np.abs(lls[i] - lls[i-1]) < tol:
                break


    plt.plot(lls)
    plt.title('Log-likelihood')
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.25)
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)

    return R
    


def main():

    D = 2 # so we can visualize it more easily
    s = 5 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 2000 # number of samples
    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2.0 + mu1
    X[1200:1800, :] = np.random.randn(600, D) + mu2
    X[1800:, :] = np.random.randn(200, D)*0.5 + mu3
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.show()

    K = 3
    gmm(X, K)


if __name__ == '__main__':
    main()