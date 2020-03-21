import numpy as np
import numpy.linalg as la
import scipy.signal as scisig
import time

"""
https://github.com/jucaleb4/My_Work/blob/master/conv/Convolution/Winograd-Convolution.ipynb read before
"""


np.set_printoptions(linewidth=100)

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

### F(4x4,3x3)

n = 2**10

g = np.random.random(3)
d = np.random.random(n)
y = np.zeros(0)

d2 = np.append(np.zeros(2),d)
d2 = np.append(d2, np.zeros(2))
for i in range(len(d2)//2-1):
    yTemp = np.dot(A23.T, (G23 @ g) * (B23.T @ d2[i*2:(i*2)+4]) )
    y = np.append(y,yTemp)

convLib = np.convolve(d,g[::-1])

print("Error:",la.norm(y-convLib)/la.norm(convLib))


# assume N=1 image, K=1 filter, C=1 channel
def simpleWinogradAlg(g,d,m,B,G,A):
    N = K = C = 1
    """
    @g: 2d.numpy_array as square filter
    @d: 2d.numpy_array as data
    @m: int as output of FIR filter F(m,r)
    """
    h,w = d.shape
    r = g.shape[0]

    assert(g.shape[0] == g.shape[1])
    assert(h%m == 0 and w%m == 0)

    h-=m; w-=m

    P = (h//m)*(w//m) # num of tiles
    a = m+r-1 # input tile size

    dChunks = np.zeros((C,P,a,a))
    for c in range(C):
        for y in range(h//m):
            for x in range(w//m):
                b = y*(w//m) + x
                dChunks[c,b] = d[(y*m):(y*m)+a, (x*m):(x*m)+a]

    print(a,K,C)
    U = np.zeros((a,a,K,C))
    for k in range(K):
        for c in range(C):
            uInterm = np.dot(G, np.dot(g, G.T))
            for e in range(a):
                for v in range(a):
                    U[e,v,k,c] = uInterm[e,v]

    print(a,C,P)
    V = np.zeros((a,a,C,P))
    for b in range(P):
        for c in range(C):
            vInterm = np.dot(B.T, np.dot(dChunks[c,b], B))
            for e in range(a):
                for v in range(a):
                    V[e,v,c,b] = vInterm[e,v]

    M = np.zeros((a,a,K,P))
    for e in range(a):
        for v in range(a):
            M[e,v] = np.dot(U[e,v], V[e,v])

    Y = np.zeros((K,P,m,m))
    for k in range(K):
        for b in range(P):
            mInterm = np.zeros((a,a))
            for e in range(a):
                for v in range(a):
                    mInterm[e,v] = M[e,v,k,b]
            Y[k,b] = np.dot(A.T, np.dot(mInterm, A))

    Ynew = np.zeros((K,h,w))
    for k in range(K):
        for y in range(h//m):
            for x in range(w//m):
                b = y*(w//m) + x
                Ynew[k,y*m:(y+1)*m, x*m:(x+1)*m] = Y[k,b]
    return Ynew

def padImage(g,r):
    h,w = g.shape
    g2 = np.zeros((2*r-2 + h,2*r-2 + w))
    g2[r-1:r-1+h,r-1:r-1+w] = g
    return g2

def revMatrix(M):
    n1,n2 = M.shape
    return np.eye(n1)[::-1] @ M @ np.eye(n2)[::-1]


n = 2**7
r = 3
f = np.random.random((r,r))
g = np.random.random((n,n))
start = time.time()
c= scisig.convolve2d(f,g)
end = time.time()
print("Time elapsed 1D: ",end-start)


g2 = revMatrix(g)
g2 = padImage(g2,r)
startWino = time.time()
cWino = simpleWinogradAlg(f,g2,2,B23,G23,A23)[0]
endWino = time.time()
print("Time elapsed 2D : ",endWino-startWino)
cWino = revMatrix(cWino)

print("Error:",la.norm(c - cWino)/la.norm(c))



def simpleWinogradAlg_FLOPS(h,w,r,m,B,G,A,N,K,C,matmul):
    assert(h%m == 0 and w%m == 0)

    h-=2; w-=2

    P = (h//m)*(w//m) # num of tiles
    a = m+r-1 # input tile size

    dChunks = np.zeros((C,P,a,a))

    flops = np.zeros(2)

    U = np.zeros((a,a,K,C))
    g = np.zeros((r,r))
    temp = K * C * ( matmul(G,g) + matmul(g,G.T))
    flops += temp

    V = np.zeros((a,a,C,P))
    temp = N * P * C * ( matmul(B.T, dChunks[0,0]) + matmul(dChunks[0,0],B) )
    flops += temp

    M = np.zeros((a,a,K,P))
    # (K,C) x (C,P)
    temp = N * a * a * matmul(U[0,0],V[0,0])
    flops += temp

    Y = np.zeros((K,P,m,m))
    mInterm = np.zeros((a,a))
    temp = K * P * ( matmul(A.T, mInterm) + matmul(mInterm, A) )
    flops += temp

    # reorder
    return flops

def directMatmul(M1,M2):
    assert(M1.shape[1] == M2.shape[0])
    return M1.shape[0] * M1.shape[1] * M2.shape[1]

def direct2DConvFlops(C,N,K,H,W,R):
    return np.asarray([C*N*K*H*W*(R**2),2*C*N*K*H*W*(R**2)])

N = 1
K = 96 # higher means more savings for multiplications
C = 3

R = 11
M = 2
p = 8
H = M**p
W = M**p

a = M + R - 1
B = np.zeros((a,a)).T
G = np.zeros((a,R))
A = np.zeros((M,a)).T
# h,w,r,m,B,G,A,N,K,C,matmul
winoFlops = simpleWinogradAlg_FLOPS(H,W,R,M,B,G,A,N,K,C,directMatmul)
# C,N,K,H,W,R
directFlops = direct2DConvFlops(C,N,K,H,W,R)

lowest = H*W*K*N*C

print("Direct Flops:",directFlops)
print("Winograd Flops:",winoFlops)

print("Winograd Savings:",directFlops/winoFlops)
