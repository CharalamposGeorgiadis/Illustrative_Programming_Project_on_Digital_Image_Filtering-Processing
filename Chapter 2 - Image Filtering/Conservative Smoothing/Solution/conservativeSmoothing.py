from Noise_Functions.gaussian import gaussian
from Noise_Functions.saltAndPepper import saltAndPepper
from Noise_Functions.whiteMultiplicative import multiplicative
import cv2
import numpy as np


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


# Function that performs conservative smoothing on a noise image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, noise image as a 3D array
def conservativeSmoothing(img, filter_size):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    # filter_size % 2 is used so that the function works if the user enters an even number for the filter size
    mod = filter_size % 2
    # Initializing an array that will hold the value of each pixel in the local pixel neighborhood. Its size is
    # pow(filter_size, 2) - 1 since we do not want this array to hold the pixel that will be compared to its
    # neighbors.
    elements = np.zeros(pow(filter_size, 2) - 1)
    # Applying 0-padding to the image
    padded_image = zero_padding(img, filter_size, mod)
    n = filter_size // 2
    for i in range(n, padded_image.shape[0] - n):
        for j in range(n, padded_image.shape[1] - n):
            for o in range(c):
                m = 0
                for k in range(filter_size):
                    for l in range(filter_size):
                        # Checking if the current pixel's coordinates are the same as the ones of the pixel that is
                        # about to be stored in the elements array. If they are, it will not be stored in the array,
                        # because, then, the algorithm will compare it to itself and not change its value. We would
                        # check if i == i + k - n and j == j + k -n. The if statement below is a simplified version of
                        # these comparisons.
                        if not (k == n and l == n):
                            elements[m] = padded_image[i + k - n][j + l - n][o]
                            m += 1
                # Sorting the elements array in order to compare the current pixel to the minimum and maximum values of
                # the elements array
                elements.sort()
                if padded_image[i][j][o] > elements[elements.size - 1]:
                    output[i - n][j - n][o] = elements[elements.size - 1]
                elif padded_image[i][j][o] < elements[0]:
                    output[i - n][j - n][o] = elements[0]
                else:
                    output[i - n][j - n][o] = padded_image[i][j][o]
            gif[i - n][j - n] = output[i - n][j - n]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the median filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Conservative smoothing row-by-row', frames[i].astype(np.uint8))
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

filter = int(input("Enter the size of the Conservative filter (3 creates a 3x3 filter):\n"))
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

# Applying Conservative Smoothing to the image
filtered_image, noise_image = conservativeSmoothing(noise_image, filter)

# Stacking the noise and filtered image horizontally so that they may be displayed side-by-side
display_window = np.hstack((noise_image, filtered_image))

# Displaying stacked images
cv2.imshow('Noise image (left) and filtered image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
