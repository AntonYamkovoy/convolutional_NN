## need to import numpy and cv2
## pip install numpy , pip install opencv-python

## few examples of convolution, algorithm is performed by opencv lib, it uses the same method as explained
# to us

import cv2
import numpy as np



# description
# needs to generate random image sizes within a range, and kernels

# params
# @ max height - maximim height of images
# @ max width - max width of images
# @ max kernel size

# @ outputs, structure to store images and kernels sets.

def create_image_set():

    return




# description
# calculates the average results for the given random generated (image,kernel) set

# params
# @ image set
# @ kernel set

# @ outputs the resulting approximation matrix, to be compared with toom cook algo outputs

def calc_average():

    return




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


# @ error rate, in comparison to base

def compare_results():

    return


















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
