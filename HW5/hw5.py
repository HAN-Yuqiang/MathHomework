# -*- coding: utf-8 -*-
import numpy as np
import cvxopt
import matplotlib.pyplot as plt

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma = 6.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2*(sigma**2)))


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        if C is None:
            self.C = C
        else:
            self.C = float(C)

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # kernel function
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in xrange(n_samples):
            for j in xrange(n_samples):
                kernel_matrix[i, j] = self.kernel(x[i], x[j])

        #solve by cvxopt pachakge
        # minimize    (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #             A*x = b.

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(no.zeros(n_samples))
        else:
            m1 = np.diag(np.ones(n_samples) * -1)
            m2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((m1, m2)))
            m1 = np.zeros(n_samples)
            m2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((m1, m2)))

        s = cvxopt.solvers.qp(P, q, G, h, A, b)


        # lagrange multipliers
        alphas = np.array(list(s['x']))

        sv = alphas > 1e-5
        index = np.where(sv)[0]
        self.alphas = alphas[sv]
        self.support_vector = x[sv]
        self.support_vector_y = y[sv]

        # w
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for i in range(len(self.alphas)):
                self.w += self.alphas[i] * self.support_vector_y[i] * self.support_vector[i]
        else:
            self.w = None

        # b
        self.b = 0
        for i in xrange(len(self.alphas)):
            self.b += self.support_vector_y[i]
            self.b -= np.sum(self.alphas * self.support_vector_y * kernel_matrix[index[i],sv])
        self.b /= len(self.alphas)


    def project(self, x):
        if self.w is None:
            y = np.zeros(len(x))
            for i in xrange(len(x)):
                s = 0
                for alpha, support_vector_y, support_vector in zip(self.alphas, self.support_vector_y, self.support_vector):
                    s += alpha * support_vector_y * self.kernel(x[i], support_vector)
                    y[i] = s
        else:
            y = np.dot(x, self.w)
        return y + self.b


    def predict(self, x):
        return np.sign(self.project(x))


def gen_data():
    mean1 = np.array([0, 3])
    mean2 = np.array([3, 0])
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    x1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(x2)) * -1

    x1_train = x1[:90]
    y1_train = y1[:90]
    x2_train = x2[:90]
    y2_train = y2[:90]
    x_train = np.vstack((x1_train, x2_train))
    y_train = np.hstack((y1_train, y2_train))

    x1_test = x1[90:]
    y1_test = y1[90:]
    x2_test = x2[90:]
    y2_test = y2[90:]
    x_test = np.vstack((x1_test, x2_test))
    y_test = np.hstack((y1_test, y2_test))
    return x_train, y_train, x_test, y_test

def show(X1_train, X2_train, clf):
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    # plt.scatter(clf.support_vector[:,0], clf.support_vector[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-7,7,50), np.linspace(-7,7,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.axis("tight")
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('SVM model from train data')
    plt.savefig('svm.png')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = gen_data()
    clf = SVM(gaussian_kernel, C=0.1)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    correct = np.sum(y_predict == y_test)
    print "%d out of %d predictions is correct" % (correct, len(y_predict))
    show(x_train[y_train==1], x_train[y_train==-1], clf)





