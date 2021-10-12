import numpy as np
import cv2


# Function that adds white multiplicative noise to an image
# param img: 4D array containing the original, clean image
# returns: noise image, original image as a 3D array
def multiplicative(img):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    # Generating a random noise array with Uniform distribution
    noise = np.random.normal(0, 1, img.shape).astype(np.uint8)
    # Calculating the output image.
    output = img + img * noise
    return output, img
