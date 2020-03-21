from __future__ import print_function
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint
from operator import mul
from functools import reduce
import numpy as np








def At(a,m,n):
    return Matrix(m, n, lambda i,j: a[i]**j)

def A(a,m,n):
    return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def T(a,n):
    return Matrix(Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i,j: -a[i]**n)))

def Lx(a,n):
    x=symbols('x')
    return Matrix(n, 1, lambda i,j: Poly((reduce(mul, ((x-a[k] if k!=i else 1) for k in range(0,n)), 1)).expand(basic=True), x))

def F(a,n):
    return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))

def Fdiag(a,n):
    f=F(a,n)
    return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))

def FdiagPlus1(a,n):
    f = Fdiag(a,n-1)
    f = f.col_insert(n-1, zeros(n-1,1))
    f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0)))
    return f

def L(a,n):
    lx = Lx(a,n)
    f = F(a, n)
    return Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T

def Bt(a,n):
    return L(a,n)*T(a,n)

def B(a,n):
    return Bt(a,n-1).row_insert(n-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

FractionsInG=0
FractionsInA=1
FractionsInB=2
FractionsInF=3

def cookToomFilter(a,n,r,fractionsIn=FractionsInG):
    alpha = n+r-1
    f = FdiagPlus1(a,alpha)
    if f[0,0] < 0:
        f[0,:] *= -1
    if fractionsIn == FractionsInG:
        AT = A(a,alpha,n).T
        G = (A(a,alpha,r).T/f).T
        BT = f * B(a,alpha).T
    elif fractionsIn == FractionsInA:
        BT = f * B(a,alpha).T
        G = A(a,alpha,r)
        AT = (A(a,alpha,n)).T/f
    elif fractionsIn == FractionsInB:
        AT = A(a,alpha,n).T
        G = A(a,alpha,r)
        BT = B(a,alpha).T
    else:
        AT = A(a,alpha,n).T
        G = A(a,alpha,r)
        BT = f * B(a,alpha).T
    return (AT,G,BT,f)


def filterVerify(n, r, AT, G, BT):

    alpha = n+r-1

    di = IndexedBase('d')
    gi = IndexedBase('g')
    d = Matrix(alpha, 1, lambda i,j: di[i])
    g = Matrix(r, 1, lambda i,j: gi[i])

    V = BT*d
    U = G*g
    M = U.multiply_elementwise(V)
    Y = simplify(AT*M)

    return Y

def convolutionVerify(n, r, B, G, A):

    di = IndexedBase('d')
    gi = IndexedBase('g')

    d = Matrix(n, 1, lambda i,j: di[i])
    g = Matrix(r, 1, lambda i,j: gi[i])

    V = A*d
    U = G*g
    M = U.multiply_elementwise(V)
    Y = simplify(B*M)

    return Y

def showCookToomFilter(a,n,r,fractionsIn=FractionsInG):

    AT,G,BT,f = cookToomFilter(a,n,r,fractionsIn)

    print ("AT = ")
    pprint(AT)
    print ("")

    print ("G = ")
    pprint(G)
    print ("")

    print ("BT = ")
    pprint(BT)
    print ("")

    if fractionsIn != FractionsInF:
        print ("FIR filter: AT*((G*g)(BT*d)) =")
        pprint(filterVerify(n,r,AT,G,BT))
        print ("")

    if fractionsIn == FractionsInF:
        print ("fractions = ")
        pprint(f)
        print ("")

def showCookToomConvolution(a,n,r,fractionsIn=FractionsInG):

    AT,G,BT,f = cookToomFilter(a,n,r,fractionsIn)

    B = BT.transpose()
    A = AT.transpose()

    calculateError(AT,G,BT,f)
    #print ("A = ")
    #pprint(A)
    #print ("")

    #print ("G = ")
    #pprint(G)
    #print ("")

    #print ("B = ")
    #pprint(B)
    #print ("")

    if fractionsIn != FractionsInF:
    	temp=2
        #print ("Linear Convolution: B*((G*g)(A*d)) =")
        #pprint(convolutionVerify(n,r,B,G,A))
        #print ("")
        


    if fractionsIn == FractionsInF:
    	temp=2
        #print ("fractions = ")
        #pprint(f)
        #print ("")

# testing method we found in the paper
H = np.array([1,2])
X = np.array([1, 1, 1])


AT,G,BT,f = cookToomFilter((0,1,-1), 2, 3)
print(AT," AT\n")
print(G," G\n")
print(BT," BT\n")
print(f," f\n")

# trying to apply method
# H = image
# X = kernel
# 1D convolution example
"""

(h ∗ x)1D = AT(Gh o BT x)



"""
tc_convolution_result = np.matmul(AT,np.multiply( np.matmul(G,H) , np.matmul(BT,X) ) )

print(tc_convolution_result)


def calculateError(point1, point2, point3):

	A,G,B,f=  cookToomFilter((0,point1, point2,point3,-1), 2, 3)

	A23 = np.asarray(A)
	B23 = np.asarray(B)
	G23 = np.asarray(G, dtype=np.int32)


	A23 = np.asarray(np.transpose(A23), dtype=np.int32)
	B23 = np.asarray(np.transpose(B23), dtype=np.int32)



	print("B23: ",B23,"\n")
	print("G23: ",G23,"\n")
	print("A23: ",A23,"\n")

	#A23 = np.asarray(np.transpose(A))
	#B23 = np.asarray(np.transpose(B))
	#G23 = np.asarray(G)




	#------------------------------------------

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

	print("Standard Error:",la.norm(convLib - conv2d, ord=2)/la.norm(convLib, ord=2))
	print("Winograd/TC Error:",la.norm(convLib - cWino, ord=2)/la.norm(convLib, ord=2))    

def runBruteForce():
	imageSize = int(input("Image Size?: "))
	kernalSize = int(input("Kernel Size?: "))
	amountOfPoints = (kernalSize + imageSize) -2

	if amountOfPoints == 3: 
		for x in range(-1000, 1000, 50):
			point1 = x/1000 
			for y in range(-1000, 1000, 100):
				point2 = y/1000 
				for z in range(-1000, 1000, 100):
					point3 = z/1000 	
					if point1 == point2 or point1 == point3 or point2 == point3 or point3 ==0 or point2 ==0 or point1 ==0 or point3 ==-1 or point2 ==-1 or point1 ==-1:
						print("Skipped points ", point1)
					else:	
						print(point1, point2, point3)
						calculateError(point1, point2, point3)
	else: print("Didn't seem to be a valid size. Sorry")	
