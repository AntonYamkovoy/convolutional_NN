import numpy as np
import numpy.linalg as la
import scipy.signal as scisig

from func import *

B23 = np.asarray([
    [1, 0,-1, 0],
    [0, 1, 1, 0],
    [0,-1, 1, 0],
    [0, 1, 0,-1]
]).T

G23 = np.asarray([
    [ 1,  0, 0],
    [.5, .5,.5],
    [.5,-.5,.5],
    [ 0, 0,  1]
])

A23 = np.asarray([
    [1,1,1,0],
    [0,1,-1,-1]
]).T


#1d case

d = np.random.random(4)
g = np.random.random(3)

inner = np.dot(G23,g) * np.dot(B23.T,d)
y = np.dot(A23.T, inner)

direct = np.asarray([sum(d[:3]*g), sum(d[1:]*g)])

print("Error:",la.norm(direct - y)/la.norm(direct))


#2d case


g = np.random.random((4,4))
f = np.random.random((3,3))

direct = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        direct[i,j] = np.sum(f * g[i:i+3,j:j+3])

inner = np.dot(G23, np.dot(f, G23.T)) * np.dot(B23.T, np.dot(g, B23))
Y = np.dot(A23.T, np.dot(inner, A23))

print("Error of one Winograd",la.norm(Y - direct)/la.norm(direct))



convLib = scisig.convolve2d(f,g)
conv2d = convolve2DToeplitz(f,g)
g2 = revMatrix(g)
g2 = padImage(g2,len(f))
cWino = simpleWinogradAlg(f,g2,2,B23,G23,A23)[0]
cWino = revMatrix(cWino)

print("Error:",la.norm(convLib - conv2d, ord=2)/la.norm(convLib, ord=2))
print("Error:",la.norm(convLib - cWino, ord=2)/la.norm(convLib, ord=2))




d = np.random.random(4)
g = np.random.random(3)
g2 = np.append(np.zeros(1),g)
direct = np.dot(A23.T,  (np.dot(G23,g) * np.dot(B23.T,d))  )
print(direct)

H = toeplitzToHankle(vectorToToeplitz(d))[2:4]
conv = np.dot(H,g2)
print(conv)



# toom cook form
convTC = monomialTC(d,g2[::-1])

print(convTC[1],"\n")
print("2nd to 4th terms:",convTC[1][2:4])
