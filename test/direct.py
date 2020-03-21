import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.signal as scisig

from func import *

np.set_printoptions(linewidth=150)


p = 1; n = 2**p

x = randomvec(n)
hvec = randomvec(n,seedNumber=p)
T = vectorToToeplitz(hvec)

toeplitzConv = np.dot(T,x)
regConv = np.convolve(x,hvec)

print(x,hvec)
print(regConv)
print("Error:",la.norm(toeplitzConv - regConv)/la.norm(regConv))



""" read this please before https://github.com/jucaleb4/My_Work/blob/master/conv/Convolution/1-and-2D-Convolution.ipynb"""

"""
1D and 2D examples of direction convolution

where X is the image, and H is the kernel


"""
print("\n")

X = np.asarray([
    [1,4,1],
    [2,5,3],
    [7,2,4]
])

H = np.asarray([
    [1,1],
    [1,-1]
])

m1,n1 = X.shape
m2,n2 = H.shape
f1,g1 = (m1+m2-1),(n1+n2-1)
print("Dimensions of both our image and kernel\n")
print(m1,n1,"|",m2,n2)
print("the resulting size of the convolution")
print(f1,g1)




H1 = vectorToToeplitz(N=3,v=np.asarray([1,-1]))[:4]
H2 = vectorToToeplitz(N=3,v=np.asarray([1, 1]))[:4]
H3 = vectorToToeplitz(N=3,v=np.asarray([0, 0]))[:4]
H4 = vectorToToeplitz(N=3,v=np.asarray([0, 0]))[:4]
Hlist = np.asarray([H1,H2,H3,H4])

Hfull = np.zeros((f1*g1,m1*n1))
for i in range(f1):
    for j in range(n1):
        Hfull[i*f1:(i+1)*f1, j*n1:(j+1)*n1] = Hlist[(i+(f1-1)*j)%f1]

print("H =\n",Hfull)



# create vectorized matrix
def quasiVec(V, blkSize):
    m,n = V.shape
    return np.reshape(V[::-1], newshape=(m*n,))

x = quasiVec(X,m1)
print(x)


def vecToConvMatrix(v, shape):
    assert(len(shape) == 2)
    return np.reshape(v, newshape = shape)[::-1]

def createDoublyToeplitz(F,fm,fn,gm,gn):
    m = fm+gm-1; n = fn+gn-1

    F2 = np.zeros((m,n)).copy()
    F2[m-fm:,:fn] = F.copy()
    F = F2

    Fs = np.zeros((m,n,gn))
    diffZero = n-fn

    Ts = np.zeros((m,n,gn))
    for y in range(m):
        smallT = vectorToToeplitz(F[y])[:n,:gn]
        Ts[m-y-1] = smallT.copy()

    Tfull = np.zeros((m*n,gm*gn))
    for i in range(m):
        sel = i
        for j in range(gm):
            Tfull[i*n:(i+1)*n, j*gn:(j+1)*gn] = Ts[sel]
            sel -= 1
            if(sel < 0):
                sel = m-1

    return Tfull

def convolve2DToeplitz(F,G):
    fm,fn = F.shape; gm,gn = G.shape
    Tfull = createDoublyToeplitz(F,fm,fn,gm,gn)
    g = quasiVec(G,gm)
    vecTConv = np.dot(Tfull, g)
    m = fm+gm-1; n = fn+gn-1
    return vecToConvMatrix(vecTConv, (m,n))


vecTConv = np.dot(Hfull,x)
toeConv = vecToConvMatrix(vecTConv, (f1,g1))
regConv = scisig.convolve2d(H,X)
automatedToeConv = convolve2DToeplitz(H,X)

print("Specific Error:",la.norm(regConv - toeConv, ord=2)/la.norm(regConv, ord=2))
print("General Algorithm Error:",la.norm(regConv - automatedToeConv, ord=2)/la.norm(regConv, ord=2))


fm,fn,gm,gn = np.random.randint(2,25,size=4)
fm = fn # square filter
f = np.random.random((fm,fn))
g = np.random.random((gm,gn))

convLib = scisig.convolve2d(f,g)
conv2d = convolve2DToeplitz(f,g)



print("Error:",la.norm(convLib - conv2d,ord=2)/la.norm(convLib,ord=2))
