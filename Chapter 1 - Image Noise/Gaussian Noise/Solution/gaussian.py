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
    # Displaying the noise signal
    cv2.imshow('Noise signal', gauss)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Clipping the output image in the [0, 1] range
    output = np.clip(output, 0.0, 1.0)
    return output * 255, img


# Reading the image
image = cv2.imread('Inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (600, 480))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 480, 600, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 480, 600, 3))

# The user enters the mean and variance of the Gaussian distribution
mean = float(input("Enter the mean of the Gaussian distribution:\n"))
variance = float(input("Enter the variance of the Gaussian distribution (must be greater than 0):\n"))

# Applying noise to the image
gaussian_image, image = gaussian(image, mean, variance)

# Stacking the clean and noise image horizontally so that they may be displayed side-by-side
display_window = np.hstack((image, gaussian_image))

# Displaying stacked images
cv2.imshow('Original image (left) and noise image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
