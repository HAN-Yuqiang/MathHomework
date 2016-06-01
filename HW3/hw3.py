# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def generate():
    '''
    this function generates three clusters in 2-D dimension
    from multivariate normal distribution
    '''
    mean1 = [3, 0]
    cov1 = np.eye(2)
    cluster1 = np.random.multivariate_normal(mean1, cov1, 100)
    x1, y1 = cluster1.T
    mean2 = [4, 5]
    cov2 = np.eye(2)
    cluster2 = np.random.multivariate_normal(mean2, cov2, 100)
    x2, y2 = cluster2.T
    mean3 = [-5, -6]
    cov3 = np.eye(2)
    cluster3 = np.random.multivariate_normal(mean3, cov3, 100)
    x3, y3 = cluster3.T
    data = np.vstack((cluster1, cluster2, cluster3))
    #save
    np.savetxt('data.txt', data, fmt='%.8f', delimiter = '\t')
    #show
    fig = plt.figure()
    plt.plot(x1, y1, '.')
    plt.plot(x2, y2, '.')
    plt.plot(x3, y3, '.')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    fig.savefig('data.png')
    # plt.show()
    return data


class mog:
    def __init__(self, x, dim, k, lambda_):
        self.x = np.mat(x)
        self.dim = dim
        self.k = k
        self.lambda_ = lambda_

    def train(self):
        # choose k points as center randomly
        center_index = np.random.choice(self.x.shape[0], self.k)
        self.centers = self.x[center_index, :]

        # compute the distance and label
        distance = np.mat(np.zeros((self.x.shape[0], self.k)))
        for i in range(self.k):
            distance[:, i] = np.sum(np.power(self.x - self.centers[i, :], 2), axis=1)
        self.label = distance.argmin(axis=1)
        self.label = np.array(self.label).flatten()

        #initialize parameters
        self.mean = self.centers
        self.sigma = np.zeros((self.x.shape[1], self.x.shape[1], self.k))
        self.prior = np.zeros((1, self.k))
        for i in range(self.k):
            cluster_i_set = self.x[self.label == i]
            self.sigma[:, :, i] = np.cov(cluster_i_set.T)
            self.prior[0 ,i] = 1.0 * cluster_i_set.shape[0] / self.x.shape[0]

        # EM algorithm
        probability = self.__EM()

        self.label = probability.argmax(axis=1)
        self.label = np.array(self.label).flatten()

    def __EM(self):
        old_log_like = -np.inf
        threshold = 1e-15
        probability = 0
        while True:
            # E step
            probability = self.__probability()
            expectation = np.multiply(probability, self.prior)
            expectation = np.divide(expectation, expectation.sum(axis=1))

            # M step: updata parameters
            sumk = expectation.sum(axis=0)
            self.prior = sumk / self.x.shape[0]
            self.mean = np.diag(np.array(np.divide(1, sumk)).flatten()) * \
                        expectation.T * self.x
            for i in range(self.k):
                x_shift = self.x - self.mean[i, :]
                self.sigma[:, :, i] = x_shift.T * \
                    np.diag(np.array(expectation[:, i]).flatten()) * x_shift /\
                    sumk[0, i]

            new_log_like = np.log(probability * self.prior.T).sum()
            if np.abs(new_log_like - old_log_like) < threshold:
                break
            old_log_like = new_log_like
        return probability

    def __probability(self):
        probability = np.mat(np.zeros((self.x.shape[0], self.k)))
        for i in range(self.k):
            x_shift = x - self.mean[i, :]
            exp_item = np.diag(x_shift * \
                                np.mat(self.sigma[:, :, i]).I * \
                                x_shift.T)
            coef = (2*np.pi) ** (-self.x.shape[1]/2) / \
                    np.sqrt(np.linalg.det(self.sigma[:, :, i]))

            probability[:, i] = coef * np.exp(-0.5*exp_item).reshape((self.x.shape[0], 1))
        return probability

    def show(self):
        fig = plt.figure()
        cluster0 = self.x[self.label == 0]
        cluster1 = self.x[self.label == 1]
        cluster2 = self.x[self.label == 2]
        plt.plot(cluster0[:,0], cluster0[:,1], '.')
        plt.plot(cluster1[:,0], cluster1[:,1], '.')
        plt.plot(cluster2[:,0], cluster2[:,1], '.')
        fig.savefig('mog.png')
        plt.show()

if __name__ == '__main__':
    x = generate()
    gmm = mog(x, 2, 3, 0)
    gmm.train()
    gmm.show()







