import numpy as np
import numpy.linalg as la
import scipy.signal as scisig
import random
from func import *
import time

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

    A,G,B,f2=  cookToomFilter(tuple(points), image_size, kernel_size)

    #print(B, "\n\n", G, "\n\n", A )

    Atemp = np.array(A, dtype=np.float32)
    Btemp = np.array(B, dtype=np.float32)
    Gtemp = np.array(G, dtype=np.float32)
    #print("Broken: ",B23, "\n\n", G23, "\n\n", A23 )
    B23 = np.asarray(Btemp).T

    G23 = np.asarray(Gtemp)

    A23 = np.asarray(Atemp).T

    #print("Working: ",B23, "\n\n", G23, "\n\n", A23 )

    g = image
    f = kernel

    direct = np.zeros((image_size,image_size))
    for i in range(image_size):
        for j in range(image_size):
            direct[i,j] = np.sum(f * g[i:i+kernel_size,j:j+kernel_size])

    inner = np.dot(G23, np.dot(f, G23.T)) * np.dot(B23.T, np.dot(g, B23))
    Y = np.dot(A23.T, np.dot(inner, A23))
    error = la.norm(Y - direct)/la.norm(direct)
    #print("Error of one Winograd",la.norm(Y - direct)/la.norm(direct))

    return error



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

        error_rate_sum += test_points_variable(points,image,kernel,image_size,kernel_size)

    return error_rate_sum/ len(imageKernelList)


def find_error_rate(image_size,kernel_size,number_of_images, number_points_sets,points_per_set,input_size):
    dictionary = {}
    resultList = []
    point_coefficient = 0
    imageKernelList = generate_set(input_size,kernel_size,number_of_images) #  eg generating 4x4 images, 3x3 kernels, 10 images in the set
    for i in range(number_points_sets):
        # loop around this n times generating a new points set each loop to test
        upper = 1
        lower = -1
        pointsSet = sample_floats(lower,upper,image_size+kernel_size-2)
        average_error = test_points_for_image_list(pointsSet,imageKernelList,image_size,kernel_size)
        print("average_error :",average_error, "@ points :",pointsSet)
        dictionary[tuple(pointsSet)] = average_error
        point_coefficient += average_error

    # exit loop now we have all average error rates for the points sets we have tested
    sorted_d = sorted((value, key) for (key,value) in dictionary.items())

    point_coefficient = point_coefficient/input_size
    #print("Point coef: ",point_coefficient)
    return sorted_d





def refine_search(high_performance_points, number_of_images, image_size, kernel_size, points_per_set):
    refined = []
    for x in range(len(high_performance_points)):
        input_size = image_size + kernel_size -1
        imageKernelList = generate_set(input_size,kernel_size,number_of_images)
        points = high_performance_points[x][1]

        refined_error = test_points_for_image_list(points, imageKernelList, image_size, kernel_size)
        print("refined_error (average of ",number_of_images," runs) + points : ",(refined_error,points))
        refined.append((refined_error,points))

    return refined


""" one thing i notice with the results is that the best points always have a split on pos/negative numbers, never pos, pos, pos || neg, neg, neg"""


# testing get error rate
image_size = 2
kernel_size = 3
number_of_images = 10
number_points_sets = 100000
points_per_set = 3
input_size = image_size + kernel_size -1 # 4
st = time.time()
result = find_error_rate(image_size,kernel_size,number_of_images, number_points_sets,points_per_set,input_size)
end = time.time()
print("Time elapsed to do initial error rate calculations: ",end-st,"s")

print("Lowest error rate points (100) :")
i=0
for i in range(100):
    print(result[i])

high_performance_points = []
for x in range(100):
    high_performance_points.append(result[x])

number_images_refined = 5000
print("Refining top ",len(high_performance_points), " points with ",number_images_refined," runs each")

refined = refine_search(high_performance_points, number_images_refined, image_size, kernel_size, points_per_set)
refined_sorted = sorted(refined)
for x in refined_sorted:
    print("refined_average_error + points", x)


"""

# testing get error rate
image_size = 4
kernel_size = 3
number_of_images = 100
number_points_sets = 1000
points_per_set = 5
input_size = image_size + kernel_size -1 # 4
st = time.time()
result = find_error_rate(image_size,kernel_size,number_of_images, number_points_sets,points_per_set,input_size)
end = time.time()
print("Time elapsed to do initial error rate calculations: ",end-st,"s")

print("Lowest error rate points (100) :")
i=0
for i in range(100):
    print(result[i])

high_performance_points = []
for x in range(100):
    high_performance_points.append(result[x])

number_images_refined = 2000
print("Refining top ",len(high_performance_points), " points with ",number_images_refined," runs each")

refined = refine_search(high_performance_points, number_images_refined, image_size, kernel_size, points_per_set)
refined_sorted = sorted(refined)
for x in refined_sorted:
    print("refined_average_error + points", x)
"""

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
imageKernelList = generate_set(4,3,100) # 4x4 images, 3x3 kernels, 2 tuples total
points = [0.8095133296178773,0.10362463148985646,0.7774840747025125]
average_error = test_points_for_image_list(points,imageKernelList,image_size,kernel_size)
print(average_error)

"""





"""
#
# TESTING VARIOUS INPUT SIZED EG F(2,3)..
#

print("f(2,3)")
image_size = 2
kernel_size = 3
points = [10,20,30]
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))
error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ",error)



print("f(2,3)")
image_size = 2
kernel_size = 3
points = [0,1,-1]
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ",error)





print("f(4,3)")
image_size = 4
kernel_size = 3
points = [-1,-0.5,0,0.5,1] # number of points = (4+3)-2 = 5
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ",error)




print("f(6,3)")
image_size = 6
kernel_size = 3
points = [-1.5,-1,-0.5,0,0.5,1,1.5] # number of points = (6+3)-2 = 7 points
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ", error)



print("f(5,3)")
image_size = 5
kernel_size = 3
points = [-1.5,-1,-0.5,0,0.5,1] # number of points = (5+3)-2 = 6 points
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ", error)


print("f(3,3)")
image_size = 3
kernel_size = 3
points = [-1,-0.5,0,0.5] # number of points = (3+3)-2 = 4 points
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ", error)


print("f(2,2)")
image_size = 2
kernel_size = 2
points = [0.5,-0.5] # number of points = (2+2)-2 = 2 points
image = np.random.random((image_size+kernel_size-1,image_size+kernel_size-1))
kernel =  np.random.random((kernel_size,kernel_size))

error = test_points_variable(points,image,kernel,image_size,kernel_size)
print("Error: ", error)

"""
