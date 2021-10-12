import numpy as np
import cv2
from Noise_Functions.gaussian import gaussian
from Noise_Functions.saltAndPepper import saltAndPepper
from Noise_Functions.whiteMultiplicative import multiplicative


# Function that applies 0-padding on an image
# param img: 3D array containing an image
# param f: size of the filter (is 3 if the filter is 3x3)
# param m: result of f % 2, used so that the function works for even-sized filters
# returns: 0-padded image
def zero_padding(img, f, m):
    h, w, c = img.shape
    padded_image = np.zeros((h + f - m, w + f - m, c))
    n = f // 2
    padded_image[n:padded_image.shape[0] - n, n:padded_image.shape[1] - n, :] = img
    return padded_image


# Function that performs standard median filtering on an image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, input image as a 3D array
def medianFiltering(img, filter_size):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    # filter_size % 2 is used so that the function works if the user enters an even number for the filter size
    mod = filter_size % 2
    # Initializing an array that will hold the value of each pixel in the local pixel neighborhood
    elements = np.zeros(pow(filter_size, 2))
    # Applying 0-padding to the image
    padded_image = zero_padding(img, filter_size, mod)
    n = filter_size // 2
    for i in range(n, padded_image.shape[0] - n):
        for j in range(n, padded_image.shape[1] - n):
            for o in range(c):
                for k in range(filter_size):
                    for l in range(filter_size):
                        # Adding the value of each pixel in the current window to the elements array
                        elements[filter_size * k + l] = padded_image[i + k - n][j + l - n][o]
                # Sorting the elements array in order to pick the median element
                elements.sort()
                # Picking the median element of the elements array
                # If the size of the filter is even, the median element is the average of the two middle elements
                if mod == 1:
                    output[i - n][j - n][o] = elements[pow(filter_size, 2) // 2]
                else:
                    output[i - n][j - n][o] = (elements[pow(filter_size, 2) // 2] +
                                               elements[(pow(filter_size, 2) // 2) - 1]) // 2
            gif[i - n][j - n] = output[i - n][j - n]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the median filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Median filter row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (450, 360))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 360, 450, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 360, 450, 3))

filter = int(input("Enter the size of the Median filter (3 creates a 3x3 filter):\n"))
while True:
    print("Enter 1 to corrupt the image with Gaussian noise:")
    print("Enter 2 to corrupt the image with Salt and Pepper noise:")
    print("Enter 3 to corrupt the image with Multiplicative noise:")
    choice = int(input())
    if choice == 1:
        mean = float(input('Enter the mean of the Gaussian distribution:\n'))
        variance = float(input('Enter the variance of the Gaussian distribution (must be greater that 0):\n'))
        # Applying Gaussian noise to the image
        noise_image, image = gaussian(image, mean, variance)
        break
    elif choice == 2:
        prob = float(input("Enter the probability for Salt and Pepper noise (must be in the (0,1) range):\n"))
        # Applying Salt and Pepper noise to the image
        noise_image, image = saltAndPepper(image, prob)
        break
    elif choice == 3:
        # Applying Multiplicative noise to the image
        noise_image, image = multiplicative(image)
        break
    else:
        print("Enter an appropriate number!\n")

# Converting the noise image to a 4D one
noise_image = np.reshape(noise_image, (1, noise_image.shape[0], noise_image.shape[1], noise_image.shape[2]))

# Applying Median filtering to the image
filtered_image, noise_image = medianFiltering(noise_image, filter)

# Stacking the noise and filtered image horizontally so that they may be displayed side-by-side
display_window = np.hstack((noise_image, filtered_image))

# Displaying stacked images
cv2.imshow('Noise image (left) and filtered image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
