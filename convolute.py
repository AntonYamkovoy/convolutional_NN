## need to import numpy and cv2
## pip install numpy , pip install opencv-python

## few examples of convolution, algorithm is performed by opencv lib, it uses the same method as explained
# to us

import cv2
import numpy as np

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
