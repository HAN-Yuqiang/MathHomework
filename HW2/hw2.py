# -*- coding: utf-8 -*-
from pylab import *

with open('optdigits.tra', 'r') as fin:
    number_threes = 0
    threes = []
    for line in fin:
        if line[-2] == '3':
            line = line[:-3].strip().split(',')
            threes.append([int(x) for x in line])
            number_threes += 1

X = np.mat(np.array(threes).T) # X.shape = (64, 389)
#normalize
mean = X.mean(1)
X = X - mean
scatter = X * X.T
U,S,VT = np.linalg.svd(scatter)
U = U[:, :2]
Y = U.T * X

location = [] #choose 5*5 points for show digit 3
for i in range(5):
    for j in range(5):
        tagX = -20 + 10 * i
        tagY = 20 - 10 * j
        this_dis = np.inf
        idx = 0
        for k in range(Y.shape[1]):
            distance = (Y[0, k] - tagX) ** 2 + (Y[1, k] - tagY) ** 2
            if distance < this_dis:
                this_dis = distance
                idx = k
        location.append(idx)

figure(figsize=(10, 6))
subplot2grid((1,3), (0, 0) , colspan = 2)
x = Y[0,:]
y = Y[1,:]
plot(x,y,'.g')
locationX = [Y[0, index] for index in location]
locationY = [Y[1, index] for index in location]
plot(locationX, locationY, 'or')
grid(color = 'grey')
xlabel('First Principle Component')
ylabel('Second Principle Component')


subplot2grid((1,3), (0, 2))
img = np.zeros((40, 40))
for i in range(len(location)):
    row = (i % 5) * 8
    col = (i / 5) * 8
    three = np.array(X[:, location[i]] + mean)
    three = three.reshape(8, 8)
    img[row : row + 8, col : col + 8] = three
imshow(img, cmap = cm.gray)
#show()
savefig('PCA_3.png')


