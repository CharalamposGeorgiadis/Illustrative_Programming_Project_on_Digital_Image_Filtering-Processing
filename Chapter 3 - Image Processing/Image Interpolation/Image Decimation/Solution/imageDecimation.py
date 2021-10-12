import numpy as np
import cv2


# Function that takes as input an image and a decimation factor and performs image decimation
# param image: original image
# param decimate_factor: decimation factor
# returns: decimated image, original image as a 3D array
def imageDecimation(img, decimation_factor):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros((h // decimation_factor, w // decimation_factor, c))
    # Placing each pixel to its corresponding position in the decimated image
    k = -1
    for i in range(0, h, decimation_factor):
        k += 1
        l = - 1
        for j in range(0, w, decimation_factor):
            l += 1
            if k < output.shape[0] and l < output.shape[1]:
                output[k][l] = img[i][j]
    return output, img


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (800, 600))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 600, 800, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 600, 800, 3))

# The user enter the decimation factor for image decimation
decimate = int(input('Enter the decimation factor (Must be a positive integer):\n'))

# Applying image decimation on the image
decimated_image, image = imageDecimation(image, decimate)

# Displaying the original image
cv2.imshow('Original image', image.astype(np.uint8))
# Displaying the decimated image
cv2.imshow('Decimated image', decimated_image.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
