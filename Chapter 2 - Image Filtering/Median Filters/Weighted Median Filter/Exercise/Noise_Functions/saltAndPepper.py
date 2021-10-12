import numpy as np
import cv2


# Function that adds salt and pepper noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def saltAndPepper(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is less than the probability, then the current pixel's value will be set to 0 (black)
            if rand < p:
                output[i][j] = 0
            # If the random number is more than 1 - p, then the current pixel's value will be set to 255 (white)
            elif rand > 1 - p:
                output[i][j] = 255
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
    return output, img


# Function that adds salt noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def salt(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is more than 1 - p, then the current pixel's value will be set to 255 (white)
            if rand > 1 - p:
                output[i][j] = 255
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
    return output, img


# Function that adds pepper noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def pepper(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is less than the probability, then the current pixel's value will be set to 0 (black)
            if rand < p:
                output[i][j] = 0
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
    return output, img
