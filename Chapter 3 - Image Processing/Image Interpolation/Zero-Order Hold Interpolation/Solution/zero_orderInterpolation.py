import numpy as np
import cv2


# Function that takes as input an image and an enlarge factor and performs Zero-Order Hold Interpolation in order to
# enlarge it
# param image: 4D array containing original image
# param enlarge_factor: enlarge factor
# returns: enlarged image, original image as a 3D array
def zero_orderInterpolation(img, enlarge_factor):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros((h * enlarge_factor, w * enlarge_factor, c))
    # Placing each pixel to its corresponding position in the enlarged image
    for i in range(h):
        i_enlarge = i * enlarge_factor
        for j in range(w):
            j_enlarge = j * enlarge_factor
            for k in range(enlarge_factor):
                for l in range(enlarge_factor):
                    output[k + i_enlarge][l + j_enlarge] = img[i][j]
    return output, img


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (320, 240))
cv2.imwrite("image.jpg", image)
# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 240, 320, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 240, 320, 3))

# The user enter the enlarge factor for image resizing
enlarge = int(input('Enter the enlarge factor (Must be a positive integer):\n'))

# Applying Zero-Order Hold Interpolation on the image
enlarged_image, image = zero_orderInterpolation(image, enlarge)

# Displaying the original image
cv2.imshow('Original image', image.astype(np.uint8))
# Displaying the enlarged image
cv2.imshow('Enlarged image', enlarged_image.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
