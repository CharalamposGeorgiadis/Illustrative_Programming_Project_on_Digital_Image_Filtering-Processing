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


# Function that performs Running Median filtering on an image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, noise image as a 3D array
def runningMedianFiltering(img, filter_size):
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
    left_col = np.zeros(filter_size)
    right_col = np.zeros(filter_size)
    for i in range(n, padded_image.shape[0] - n):
        for o in range(c):
            # Initializing histogram
            hist = np.zeros(256)
            # Calculating the histogram and the median of the first window
            for j in range(filter_size):
                for k in range(filter_size):
                    elements[filter_size * j + k] = padded_image[i + j - n][k][o]
                    grey = int(padded_image[i + j - n][k][o])
                    hist[grey] += 1
            elements.sort()
            if elements.size % 2 == 1:
                median = int(elements[pow(filter_size, 2) // 2])
            else:
                median = int(elements[pow(filter_size, 2) // 2] + elements[(pow(filter_size, 2) // 2) - 1])
            output[i - n][0][o] = median
            gif[i - n][0][o] = output[i - n][0][o]
            # Calculating ltmdn
            ltmdn = 0
            for j in range(median):
                ltmdn += hist[j]
            # Calculating the median for the rest of this line
            for j in range(n + 1, padded_image.shape[1] - n):
                # Updating histogram and calculating ltmdn
                for k in range(filter_size):
                    left_col[k] = padded_image[i + k - n][j - n - 1][o]
                    grey = int(left_col[k])
                    hist[grey] -= 1
                    if grey < median:
                        ltmdn -= 1
                    right_col[k] = padded_image[i + k - n][j + n][o]
                    grey = int(right_col[k])
                    hist[grey] += 1
                    if grey < median:
                        ltmdn += 1
                while ltmdn > pow(filter_size, 2) // 2:
                    median -= 1
                    ltmdn -= hist[median]
                while ltmdn + hist[median] <= pow(filter_size, 2) // 2:
                    ltmdn += hist[median]
                    median += 1
                output[i - n][j - n][o] = median
                gif[i - n][j - n][o] = output[i - n][j - n][o]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the Running Median filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Running Median filter row-by-row', frames[i].astype(np.uint8))
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

filter = int(input("Enter the size of the Running Median filter (3 creates a 3x3 filter):\n"))
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

# Applying Running Median filtering to the image
filtered_image, noise_image = runningMedianFiltering(noise_image, filter)

# Stacking the noise and filtered image horizontally so that they may be displayed side-by-side
display_window = np.hstack((noise_image, filtered_image))

# Displaying stacked images
cv2.imshow('Noise image (left) and filtered image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
