import numpy as np
import cv2


# Function that adds Gaussian noise to an image
# param img: 4D array containing the original, clean image
# param m: mean of the Gaussian distribution
# param v: variance of the Gaussian distribution
# returns: noise image, original image as a 3D array
def gaussian(img, m, v):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    # Calculating the sigma of the distribution (σ)
    sigma = v ** 0.5
    # Generating a random noise array in the range from the Gaussian distribution
    gauss = np.random.normal(m, sigma, img.shape)
    # The original image is divided by 255 in order to be in the same range as the gaussian array
    output = img / 255 + gauss
    # Clipping the output image in the [0, 1] range
    output = np.clip(output, 0, 1.0)
    return output * 255, img
