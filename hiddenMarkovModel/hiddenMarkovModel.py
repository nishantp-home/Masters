
import numpy as np
import matplotlib.pyplot as plt

def randomNormalized(d1, d2):
    """Creates a valid random Markov matrix of dimensions d1 X d2"""
    x = np.random.random((d1, d2))             
    return x / x.sum(axis=1, keepdims=True)     # Making sure that all the rows sum to 1


class HMM:
    def __init__(self, hiddenStateCount) -> None:
        self.hiddenStateCount = hiddenStateCount        # M : Number of hidden states

    def fit(self, X, maxIter=30):
        np.random.seed(123)   # Seed for testing algorithm and verifying

        vocabularySize = max(max(x) for x in X) + 1            # input observations are numbered: 0 to 1
        sequenceCount = len(X)                                            

        self.pi = np.ones(self.hiddenStateCount) / self.hiddenStateCount        # initial state distribution
        self.A = randomNormalized(self.hiddenStateCount, self.hiddenStateCount) # state transition matrix    
        self.B = randomNormalized(self.hiddenStateCount, vocabularySize)        # output distribution

        print("Initial A:", self.A)
        print("Initial B:", self.B)

        costs = []
        for it in range(maxIter):
            if it % 10 == 0:
                print("it:", it)
            alphas = []
            betas = []
            P = np.zeros(sequenceCount)   # Probabilities
            for n in range(sequenceCount):
                x = X[n]   # nth observation
                T = len(x)
                alpha = np.zeros((T, self.hiddenStateCount))
                alpha[0] = self.pi * self.B[:, x[0]]    # First value of alpha
                for t in range(1, T):
                    tmp1 = alpha[t-1].dot(self.A) * self.B[:, x[t]]   # Element by element multiplication with B
                    alpha[t] = tmp1
                P[n] = alpha[-1].sum()     # Sum of all the last alphas
                alphas.append(alpha)

                beta = np.zeros((T, self.hiddenStateCount))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])  # Element by element multiplication with B
                betas.append(beta)

            assert(np.all(P > 0))
            cost = np.sum(np.log(P))     # Total log likelihood
            costs.append(cost)

            self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(sequenceCount)) / sequenceCount

            denominator1 = np.zeros((self.hiddenStateCount, 1))
            denominator2 = np.zeros((self.hiddenStateCount, 1))
            a_num = 0
            b_num = 0
            for n in range(sequenceCount):
                x = X[n]
                T = len(x)
                denominator1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                denominator2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

                a_num_n = np.zeros((self.hiddenStateCount, self.hiddenStateCount))
                for i in range(self.hiddenStateCount):
                    for j in range(self.hiddenStateCount):
                        for t in range(T-1):
                            a_num_n[i, j] += alphas[n][t, i] * self.A[i, j] * self.B[j, x[t+1]] * betas[n][t+1, j]
                a_num += a_num_n / P[n]

                b_num_n = np.zeros((self.hiddenStateCount, vocabularySize))
                for i in range(self.hiddenStateCount):
                    for t in range(T):
                        b_num_n[i, x[t]] += alphas[n][t, i] * betas[n][t, i]
                b_num += b_num_n / P[n]

            self.A = a_num / denominator1
            self.B = b_num / denominator2

        print("A", self.A)
        print("B:", self.B)
        print("pi:", self.pi)

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        T = len(x)
        alpha = np.zeros((T, self.hiddenStateCount))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
        return alpha[-1].sum()
    
    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])
    
    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))
    
    def get_state_sequence(self, x):
        T = len(x)
        delta = np.zeros((T, self.hiddenStateCount))
        psi = np.zeros((T, self.hiddenStateCount))
        delta[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            for j in range(self.hiddenStateCount):
                delta[t, j] = np.max(delta[t-1]*self.A[:, j]) * self.B[j, x[t]]
                psi[t, j] = np.argmax(delta[t-1]*self.A[:,j])

        #backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])    # Last state
        for t in range(T-2, -1, -1):    # Loop through rest of the times in descending order
            states[t] = psi[t+1, states[t+1]]
        return states
    
def fit_coin():
    X = []
    for line in open('hiddenMarkovModel/coin_data.txt'):
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hiddenstateCount = 2
    hmm = HMM(hiddenstateCount)
    hmm.fit(X)
    L = hmm.log_likelihood_multi(X).sum()
    print("LL with fitted params:", L)

    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmm.log_likelihood_multi(X).sum()
    print("LL with true params:", L)

    # try Viterbi
    print("Best state sequence for:", X[0])
    print(hmm.get_state_sequence(X[0]))

if __name__ == '__main__':
    fit_coin()


















            

