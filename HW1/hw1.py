# -*- coding: UTF-8 -*- 
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def residuals_func(p, x, y, regular):

	f = np.poly1d(p)
	res = f(x) - y
	if regular > 0:
		res = np.append(res, np.sqrt(regular)*p)

	return res


def curve_fitting(npts, degree, text, regular):

	pts = 1000
	x = np.linspace(0, 1, pts)
	y = np.sin(2*np.pi*x)

	nx = np.linspace(0, 1, npts)
	ny = np.sin(2*np.pi*nx) + normal(0, 0.1, npts)

	p_init = np.random.randn(degree+1)
	lsq = leastsq(residuals_func, p_init, args=(nx, ny, regular))

	# p = np.poly1d(np.polyfit(nx, ny, degree))
	p = np.poly1d(lsq[0])

	plt.plot(x, y, 'g-', nx, ny, 'bo', x, p(x), 'r-', linewidth=3)
	plt.axis([-0.05, 1.05, -1.5, 1.5])
	plt.text(0.7, 0.8, text, fontsize=16)
	plt.show()


if  __name__ == '__main__':

	npts1 = 10
	npts2 = 15
	npts3 = 100
	degree1 = 3
	degree2 = 9

	curve_fitting(npts1, degree1, 'M = 3', 0)
	# curve_fitting(npts1, degree2, 'M = 9', 0)
	# curve_fitting(npts2, degree2, 'N = 15', 0)
	# curve_fitting(npts3, degree2, 'N = 100', 0)
	# curve_fitting(npts1, degree2, 'lambda=0.0001', 0.0001)
