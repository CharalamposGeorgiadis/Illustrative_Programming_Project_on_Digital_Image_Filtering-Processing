import numpy as np
import cv2
from Noise_Functions.gaussian import gaussian
from Noise_Functions.saltAndPepper import saltAndPepper
from Noise_Functions.whiteMultiplicative import multiplicative


# Function that applies 0-padding on an image
# param img: 3D array containing an image
# param f: size of the filter (is 3 if the filter is 3x3)
# returns: 0-padded image
def zero_padding(img, f):
    h, w, c = img.shape
    # f % 2 is used so that the function works if the user enters an even number for the filter size
    mod = f % 2
    padded_image = np.zeros((h + f - mod, w + f - mod, c))
    n = f // 2
    padded_image[n:padded_image.shape[0] - n, n:padded_image.shape[1] - n, :] = img
    return padded_image


# Function that creates the Weighted Median filter kernel
# param f: the size of the Weighted Median filter kernel
# return: Weighted Median filter kernel
def createWeightKernel(f):
    weights = np.zeros((f, f), np.int)
    n = f // 2
    b = -1
    current_weight = 0
    # The modulo function is used so that the algorithm works for even-sized filters
    mod = (f + 1) % 2
    for i in range(f):
        # The first element of each row is greater than that of the previous one by 1, until we reach the middle row.
        if i <= n - mod:
            current_weight = i + 1
        else:
            # In this case mod == 1 means that the filter's size is even
            if mod == 1 and i == n:
                current_weight += 2
            # After the middle row is reached, every first element of the next rows is less than that of the previous
            # one by 1
            else:
                b += 2
                current_weight = i - 1 * b
        for j in range(f):
            weights[i][j] = current_weight
            if mod == 1 and i == n and j == 0:
                current_weight -= 2
            # Until the middle element of each row is reached, every element is by 1 greater than the previous one
            elif j < n:
                current_weight += 1
            # Once the middle has been reached, the value of each next element starts declining by 1
            else:
                current_weight -= 1
    return weights


# Function that performs Weighted Median filtering on an image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, noise image as a 3D array
def weightedMedianFiltering(img, filter_size):
    weights = createWeightKernel(filter_size)
    print("The Weighted Median filter kernel is the following: ")
    print(weights)
    input("Enter any key to continue...")
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    # Applying 0-padding to the image
    padded_image = zero_padding(img, filter_size)
    n = filter_size // 2
    # Calculating the sum of the weights
    sum_of_weights = np.sum(weights)
    # Initializing an array that will hold the value of each pixel in the local pixel neighborhood
    elements = np.zeros(int(sum_of_weights))
    # Calculating the median index of the elements array
    median_position = sum_of_weights // 2
    # Performing Weighted Median filtering if the sum of the weights is an odd number
    for i in range(n, padded_image.shape[0] - n):
        for j in range(n, padded_image.shape[1] - n):
            for o in range(c):
                m = 0
                for k in range(filter_size):
                    for l in range(filter_size):
                        for p in range(weights[k][l]):
                            # The amount of pixels that will be added to the elements array is based on the weight
                            # of the current pixel
                            elements[m] = padded_image[i + k - n][j + l - n][o]
                            m += 1
                # Sorting the elements array in order to pick the median element
                elements.sort()
                output[i - n][j - n][o] = elements[median_position]
            gif[i - n][j - n] = output[i - n][j - n]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the Weighted Median filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Weighted Median filter row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (450, 360), 0)

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 360, 450, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 360, 450, 3))

filter = int(input("Enter the size of the Weighted Median filter (3 creates a 3x3 filter):\n"))
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

# Applying Weighted Median filtering to the image
filtered_image, noise_image = weightedMedianFiltering(noise_image, filter)

# Stacking the noise and filtered image horizontally so that they may be displayed side-by-side
display_window = np.hstack((noise_image, filtered_image))

# Displaying stacked images
cv2.imshow('Noise image (left) and filtered image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
