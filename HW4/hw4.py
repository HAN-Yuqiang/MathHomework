# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return x**4 + x**3 + x**2 + 1


def gradient(x, dim, delta):
    g = np.zeros(dim)
    tempx = np.zeros(dim)

    for i in xrange(dim):
        tempx[i] = delta
        g = (function(x+tempx) - function(x-tempx)) / (2*delta)
        tempx[i] = 0
    return g


def hessian(x, dim, delta):
    h = np.zeros((dim, dim))
    tempx1 = np.zeros(dim)
    tempx2 = np.zeros(dim)

    for i in xrange(dim):
        tempx1[i] = delta
        x1 = x + tempx1
        x2 = x - tempx1

        for j in xrange(i, dim):
            tempx2[j] = delta
            d2 = (function(x2 + tempx2) - function(x1 + tempx2)) / (2*delta)
            d1 = (function(x2 - tempx2) - function(x1 - tempx2)) / (2*delta)
            h[i][j] = (d2 - d1) / (2*delta)
            h[j][i] = h[i][j]
            tempx2[j] = 0
        tempx1[i] = 0
    return h


def lm(dim, initialx, maxit = 100, stop=1e-15, delta = 0.0001):
    x = np.copy(initialx)
    x_sequence = [x]
    y_sequesnce = [function(x)]
    qs = y_sequesnce

    u = 0.001
    step = 0

    for i in xrange(maxit):

        g = gradient(x, dim, delta)
        h = hessian(x, dim, delta)
        if (g**2).sum() < stop:
            break
        singular = True
        while singular:
            try:
                s = np.linalg.solve(h+u*np.eye(h.shape[0]), g)
                singular = False
              
            except Exception, e:
                u = 4 * u
        x = x + s
        x_sequence.append(x)
        y_sequesnce.append(function(x))
     
        q = y_sequesnce[-1] + (g*s).sum() + \
            0.5 * np.mat(s) * np.mat(h) * np.mat(s).T
        qs.append(q)
        r = (y_sequesnce[i+1] - y_sequesnce[i]) / (qs[i+1] - qs[i])

        if r<=0:
            x = x -s
        if r < 0.25:
            u = 4 * u
        elif r > 0.75:
            u = 0.5 * u
        else:
            pass

    return np.array(x_sequence), y_sequesnce


if __name__ == '__main__':
    x_sequence, y_sequesnce = lm(1, [60])

    # draw the result
    print x_sequence
    fig = plt.figure()
    t = np.arange(-100.0, 100.0, 0.02)
    plt.plot(t, function(t), 'b--', label='$x^4 + x^3 + x^2 + 1$')
    plt.plot(x_sequence, function(x_sequence), 'r^-', label='x iteration')
    plt.legend()
    plt.grid(True)
    fig.savefig("lm.png")
    plt.show()


    # for latex table
    x = x_sequence.flatten()
    fin = open('x.txt', 'w')
    length = len(x)
    print length
    line = 'step&'
    for i in xrange(length):
        line += str(i) + '&'
    line = line+'\\ \hline' + '\n'
    fin.write(line)

    line = 'x&'
    for i in xrange(length):
        line += '%.4f&' %(x[i])
    line = line+'\\ \hline' + '\n'
    fin.write(line)
    fin.close()













