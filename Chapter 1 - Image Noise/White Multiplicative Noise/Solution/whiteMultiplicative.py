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
    # Displaying the noise signal
    cv2.imshow('Noise signal', noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Calculating the output image.
    output = img + img * noise
    return output, img


# Reading the image
image = cv2.imread('Inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (600, 480))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 480, 600, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 480, 600, 3))

# Applying noise to the image
multiplicative_image, image = multiplicative(image)

# Stacking the clean and noise image horizontally so that they may be displayed side-by-side
display_window = np.hstack((image, multiplicative_image))

# Displaying stacked images
cv2.imshow('Original image (left) and noise image (right)', display_window)
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
