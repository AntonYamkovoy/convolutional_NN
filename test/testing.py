import numpy as np
import numpy.linalg as la
import scipy.signal as scisig
import random
from func import *

from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint
from operator import mul
from functools import reduce


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

    print("AT = ")
    pprint(AT)
    print("")

    print("G = ")
    pprint(G)
    print("")

    print("BT = ")
    pprint(BT)
    print("")

    if fractionsIn != FractionsInF:
        print("FIR filter: AT*((G*g)(BT*d)) =")
        pprint(filterVerify(n,r,AT,G,BT))
        print("")

    if fractionsIn == FractionsInF:
        print("fractions = ")
        pprint(f)
        print("")

def showCookToomConvolution(a,n,r,fractionsIn=FractionsInG):

    AT,G,BT,f = cookToomFilter(a,n,r,fractionsIn)

    B = BT.transpose()
    A = AT.transpose()

    print("A = ")
    pprint(A)
    print("")

    print("G = ")
    pprint(G)
    print("")

    print("B = ")
    pprint(B)
    print("")

    if fractionsIn != FractionsInF:
        print("Linear Convolution: B*((G*g)(A*d)) =")
        pprint(convolutionVerify(n,r,B,G,A))
        print("")

    if fractionsIn == FractionsInF:
        print("fractions = ")
        pprint(f)
        print("")






# input size = a = m + r -1
# kernel size = r
# output size = r
# number of points = m + r - 2

# function for testing 2d convolution on a variable amount of points
def test_points_variable(points, image, kernel ,image_size, kernel_size):

    points_tuple = tuple(points)
    if len(points) > ((image_size + kernel_size ) - 2):
        return None
    # points list is of the correct size
    A,G,B,f=  cookToomFilter(points_tuple, image_size, kernel_size)

    tempA = np.asarray(A)
    tempB = np.asarray(B)
    finalG = np.asarray(G, dtype=np.int32)
    finalA = np.asarray(np.transpose(tempA), dtype=np.int32)
    finalB = np.asarray(np.transpose(tempB), dtype=np.int32)
    g = image
    f = kernel
    direct = np.zeros((image_size,image_size))
    for i in range(image_size):
        for j in range(image_size):
            direct[i,j] = np.sum(f * g[i:i+kernel_size,j:j+kernel_size])

    inner = np.dot(finalG, np.dot(f, finalG.T)) * np.dot(finalB.T, np.dot(g, finalB))
    Y = np.dot(finalA.T, np.dot(inner, finalA))

    print("Error of one Winograd",la.norm(Y - direct)/la.norm(direct))

    convLib = scisig.convolve2d(f,g)
    conv2d = convolve2DToeplitz(f,g)
    g2 = revMatrix(g)
    if image_size == 4:
        g2 = padImage(g2,len(f)-1)
    else:
        g2 = padImage(g2,len(f))


    print("g2: ",g2.shape)
    cWino = simpleWinogradAlg(f,g2,image_size,finalB,finalG,finalA)[0]
    cWino = revMatrix(cWino)
    cWino = padImage(cWino,len(f))

    print("Standard Error:",la.norm( convLib - conv2d)/la.norm(convLib))
    #final_error = la.norm(convLib - cWino)/la.norm(convLib)
    final_error =la.norm( convLib- cWino) / la.norm(convLib)
    #final_error =0
    print("Winograd/TC Error:",final_error)


    return final_error



def generate_set(image_size,kernel_size, size):
    list= []
    for i in range(size):
        image = np.random.random((image_size,image_size))
        kernel = np.random.random((kernel_size,kernel_size))
        list.append((image,kernel))

    return list

def sample_floats(low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result

# returns the averag error rate for a given set of points for a image set
def test_points_for_image_list(points, imageKernelList, image_size, kernel_size):
    error_rate_sum = 0
    for tuple in imageKernelList:
        image = tuple[0]
        kernel = tuple[1]
        pointsTuple = points

        if image_size == 2:
            error_rate_sum += test_points23(points,image,kernel,image_size,kernel_size)
        elif image_size == 4 or image_size == 6:
            error_rate_sum += test_points_variable(points,image,kernel,image_size,kernel_size)
        else:
            print("Image dimensions not supported yet")
    return error_rate_sum/ len(imageKernelList)


def find_error_rate(image_size,kernel_size,number_of_images, number_points_sets,points_per_set,input_size):
    dictionary = {}
    resultList = []
    point_coefficient = 0
    imageKernelList = generate_set(input_size,kernel_size,number_of_images) #  eg generating 4x4 images, 3x3 kernels, 10 images in the set
    for i in range(number_points_sets):
        # loop around this n times generating a new points set each loop to test
        upper = 1
        lower = 0
        pointsSet = sample_floats(lower,upper,image_size+kernel_size-2)
        average_error = test_points_for_image_list(pointsSet,imageKernelList,image_size,kernel_size)
        print("average_error :",average_error, "@ points :",pointsSet)
        dictionary[tuple(pointsSet)] = average_error
        point_coefficient += average_error

    # exit loop now we have all average error rates for the points sets we have tested
    sorted_d = sorted((value, key) for (key,value) in dictionary.items())
    """
    for key, value in sorted_d.items():
        temp = [key,value]
        resultList.append(temp)
    """
    point_coefficient = point_coefficient/input_size
    #print("Point coef: ",point_coefficient)
    return sorted_d


# input size = a = m + r -1
# kernel size = r
# output size = r
# number of points = m + r - 2

# function for testing 2d convolution on a variable amount of points
def test_points23(points, image, kernel ,image_size, kernel_size):

    points_tuple = tuple(points)
    if len(points) > ((image_size + kernel_size ) - 2):
        return None
    # points list is of the correct size
    A,G,B,f2=  cookToomFilter(points_tuple, image_size, kernel_size)

    tempA = np.asarray(A)
    tempB = np.asarray(B)
    finalG = np.asarray(G, dtype=np.int32)
    finalA = np.asarray(np.transpose(tempA), dtype=np.int32)
    finalB = np.asarray(np.transpose(tempB), dtype=np.int32)
    g = image
    f = kernel
    direct = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            direct[i,j] = np.sum(f * g[i:i+3,j:j+3])

    inner = np.dot(finalG, np.dot(f, finalG.T)) * np.dot(finalB.T, np.dot(g, finalB))
    Y = np.dot(finalA.T, np.dot(inner, finalA))

    #print("Error of one Winograd",la.norm(Y - direct)/la.norm(direct))

    convLib = scisig.convolve2d(f,g)
    conv2d = convolve2DToeplitz(f,g)
    g2 = revMatrix(g)
    g2 = padImage(g2,len(f))
    cWino = simpleWinogradAlg(f,g2,2,finalB,finalG,finalA)[0]
    cWino = revMatrix(cWino)

    #print("Error:",la.norm(convLib - conv2d, ord=2)/la.norm(convLib, ord=2))
    final_error = la.norm(convLib - cWino, ord=2)/la.norm(convLib, ord=2)
    #print("Error:",final_error)


    return final_error




# testing get error rate
image_size = 2
kernel_size = 3
number_of_images = 300
number_points_sets = 10
points_per_set = 3
input_size = image_size + kernel_size -1 # 4
result = find_error_rate(image_size,kernel_size,number_of_images, number_points_sets,points_per_set,input_size)

print("Lowest error rate points:")
i=0
for i in range(10):
    print(result[i])


"""
# generates unique floating points random list
# possible improvements later to focus only on rational numbers and to try something with reciprocal values

points = sample_floats(0,1,3)
print(points)
"""



"""

# testing average error rate function for a given points set
# also returns a value to access the effectiveness of the point picker function
# it will return the average performance of the points sets picked
image_size = 2
kernel_size = 3
imageKernelList = generate_set(4,3,10) # 4x4 images, 3x3 kernels, 2 tuples total
points = [0,1,-1]
average_error = test_points_for_image_list(points,imageKernelList,image_size,kernel_size)
print(average_error)
"""


"""
# TESTING
print("ex 23- 1")
image_size = 2
kernel_size = 3
points = [0.5,0.2,-0.1]
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))
error = test_points23(points,image,kernel,image_size,kernel_size)




print("ex 23- 2")
image_size = 2
kernel_size = 3
points = [1,0,-1]
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points23(points,image,kernel,image_size,kernel_size)
print(error)


print("ex 23- 2")
image_size = 6
kernel_size = 3
points = [-1.5,-1,-0.5,0,0.5,1,1.5] # number of points = (6+3)-2 = 7
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)



print("ex 23- 2")
image_size = 4
kernel_size = 3
points = [-1,-0.5,0,0.5,1] # number of points = (4+3)-2 = 5
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
"""




"""
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



"""

"""
# need to generate A23, G23, B23 matrices
#------------------------------------------
A,G,B,f=  cookToomFilter((0,1,-1), 2, 3)

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
"""
