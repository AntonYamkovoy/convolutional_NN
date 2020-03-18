## need to import numpy and cv2
## pip install numpy , pip install opencv-python

## few examples of convolution, algorithm is performed by opencv lib, it uses the same method as explained
# to us

import cv2
import numpy as np



# description
# needs to generate random image sizes within a range, and kernels

# params
# @ max size of image
# @ max kernel size
# @ set size of the output

# @ outputs, structure to store images and kernels sets.

def create_image_set(image_size, kernel_size, set_size):
    result = []
    for x in range(set_size):
        image = np.random.normal(-1, 1, (image_size, image_size))
        kernel = np.random.normal(-1, 1, (kernel_size, kernel_size))
        pair = (image,kernel)
        result.append(pair)

    return result




# description
# calculates the average results for the given random generated (image,kernel) set

# params
# @ image set
# @ kernel set

# @ outputs the resulting approximation matrix, to be compared with toom cook algo outputs

def calc_average(image_kernel_set):


    conv_images = []
    for tuple in image_kernel_set:
        result = cv2.filter2D(src=tuple[0], kernel=tuple[1], ddepth=-1)# did conv on given random pair
        conv_images.append(result)
    # now we have a list of conv images in the list
    if(len(conv_images)== 0):
        return None
    dst = conv_images[0]

    for i in range(len(conv_images)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(conv_images[i], alpha, dst, beta, 0.0)


    #cv2.imshow('averaged out random image+kernel convolution', dst)
    print(dst)
    print("sum matrix")
    result_matrix = np.divide(dst, len(conv_images))

    print(result_matrix)
    print("result_matrix, divided by n, where n is the set size")

    cv2.imshow('avg image', result_matrix)
    # returns the approximate result matrix (average of multiple random gen images + kernel combinations)
    return result_matrix




# description
# generates toom cook algo for fast convolution

# params
# @ points set for toom cook
# @ image set
# @ kernel set

# @ outputs a result using the generatd toom cook algo based on the points inputted

def generate_toom_cook_results():

    return


# description
# compares toom cook generated matrix to direct convolution matrix

# params
# @ toom cook results matrix
# @ direct results matrix


# @ returns the average distance / absolute difference between values in the matrix
#
def compare_results(default_matrix, result_matrix):
    # finding the sum of the differences of the corresponding matrix eelements to claculate
    # an error value based on the default result_matrix
    sum =0
    default_h = default_matrix.shape[0]
    default_w = default_matrix.shape[1]
    result_h = result_matrix.shape[0]
    result_w = result_matrix.shape[1]



    if default_h == result_h and default_w == result_w:
        # can continue given matrix dimensions are equal
        for x in range(default_h):
            for y in range(default_w):
                diff = np.absolute(default_matrix.item((x, y)) - result_matrix.item((x, y)))
                sum += diff
        return sum/(default_h * default_w)

    return None













image_set = create_image_set(3, 3, 10)
calc_average(image_set)
cv2.waitKey(0)
cv2.destroyAllWindows()


# testing diff function
"""
x= np.array([[0.2, 0.4, -0.1],
            [0.7, 0.01, 0.9],
            [1, -0.7, -0.5]])

y = np.array([[1,  0.8,  0.1],
            [0.4,  -0.5,  0.3],
            [-0.6, 0.7, 0.9]])

print("comparison")
print(compare_results(x,y))


"""

"""
# examples of what different kernels do on real images

# load the image and scale to 0..1
image_input = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

# load + show the original
cv2.imshow('original', image_input)

# horizontal edge detector
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
result = cv2.filter2D(src=image_input, kernel=kernel, ddepth=-1)
cv2.imshow('horizontal edges', result)

# vertical edge detector
kernel = np.array([[1,  1,  1],
                   [0,  0,  0],
                   [-1, -1, -1]])
result = cv2.filter2D(src=image_input, kernel=kernel, ddepth=-1)
cv2.imshow('vertical edges', result)

# blurring ("box blur", because it's a box of ones)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9.0
result = cv2.filter2D(src=image_input, kernel=kernel, ddepth=-1)
cv2.imshow('blurred', result)

# sharpening
kernel = (np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]]))
result = cv2.filter2D(src=image_input, kernel=kernel, ddepth=-1)
cv2.imshow('sharpened', result)

# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
