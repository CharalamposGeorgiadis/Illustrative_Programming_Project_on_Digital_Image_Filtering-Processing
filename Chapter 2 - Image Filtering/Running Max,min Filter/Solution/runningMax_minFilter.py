import cv2
import numpy as np
from Noise_Functions.saltAndPepper import *


# Function that performs running Max filtering on a noise image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, noise image as a 3D array
def runningMaxFiltering(img, filter_size):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    temp_output = np.zeros(img.shape)
    n = filter_size // 2
    # Calculating the max row-wise
    for i in range(img.shape[0]):
        # Calculating the first output point
        max = img[i][0]
        for j in range(1, filter_size):
            for o in range(c):
                if max[o] < img[i][j][o]:
                    max[o] = img[i][j][o]
            temp_output[i][n] = max
        # Calculating the max for the rest of this line
        for j in range(img.shape[1]):
            for o in range(c):
                if img[i][j][o] >= max[o]:
                    max[o] = img[i][j][o]
                else:
                    if img[i][j - n - 1][o] == max[o]:
                        max[o] = img[i][j - n][o]
                        for k in range(1, n + 1):
                            if max[o] < img[i][j + k - n][o]:
                                max[o] = img[i][j + k - n][o]
            temp_output[i][j] = max
    # Calculating the max column-wise
    for i in range(img.shape[1]):
        # Calculating the first output point
        max = temp_output[0][i]
        for j in range(1, filter_size):
            for o in range(c):
                if max[o] < temp_output[j][i][o]:
                    max[o] = temp_output[j][i][o]
            output[n][i] = max
        gif[n][i] = output[n][i]
        # Calculating the max for the rest of this column
        for j in range(img.shape[0]):
            for o in range(c):
                if temp_output[j][i][o] >= max[o]:
                    max[o] = temp_output[j][i][o]
                else:
                    if temp_output[j - n - 1][i][o] == max[o]:
                        max[o] = temp_output[j - n][i][o]
                        for k in range(1, n + 1):
                            if max[o] < temp_output[j + k - n][i][o]:
                                max[o] = temp_output[j + k - n][i][o]
            output[j][i] = max
            gif[j][i] = output[j][i]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the Running Max filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Running Max filter column-by-column', frames[i].astype(np.uint8))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Function that performs running min filtering on a noise image
# param img: 4D array containing a noise image
# param filter_size: size of the filter (is 3 if the filter is 3x3)
# returns: filtered image, noise image as a 3D array
def runningMinFiltering(img, filter_size):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    temp_output = np.zeros(img.shape)
    n = filter_size // 2
    # Calculating the min row-wise
    for i in range(img.shape[0]):
        # Calculating the first output point
        min = img[i][0]
        for j in range(1, filter_size):
            for o in range(c):
                if min[o] > img[i][j][o]:
                    min[o] = img[i][j][o]
            temp_output[i][n] = min
        # Calculating the min for the rest of this line
        for j in range(img.shape[1]):
            for o in range(c):
                if img[i][j][o] <= min[o]:
                    min[o] = img[i][j][o]
                else:
                    if img[i][j - n - 1][o] == min[o]:
                        min[o] = img[i][j - n][o]
                        for k in range(1, n + 1):
                            if min[o] > img[i][j + k - n][o]:
                                min[o] = img[i][j + k - n][o]
            temp_output[i][j] = min
    # Calculating the min column-wise
    for i in range(img.shape[1]):
        # Calculating the first output point
        min = temp_output[0][i]
        for j in range(1, filter_size):
            for o in range(c):
                if min[o] > temp_output[j][i][o]:
                    min[o] = temp_output[j][i][o]
            output[n][i] = min
        gif[n][i] = output[n][i]
        # Calculating the min for the rest of this column
        for j in range(img.shape[0]):
            for o in range(c):
                if temp_output[j][i][o] <= min[o]:
                    min[o] = temp_output[j][i][o]
                else:
                    if temp_output[j - n - 1][i][o] == min[o]:
                        min[o] = temp_output[j - n][i][o]
                        for k in range(1, n + 1):
                            if min[o] > temp_output[j + k - n][i][o]:
                                min[o] = temp_output[j + k - n][i][o]
            output[j][i] = min
            gif[j][i] = output[j][i]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the Running min filter de-noises the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Running Min filter column-by-column', frames[i].astype(np.uint8))
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

filter = int(input("Enter the size of the Running Max/min filter (3 creates a 3x3 filter):\n"))
while True:
    print("Enter 1 to corrupt the image with Salt and Pepper noise:")
    print("Enter 2 to corrupt the image with Salt noise:")
    print("Enter 3 to corrupt the image with Pepper noise:")
    choice = int(input())
    if choice == 1:
        prob = float(input("Enter the probability for Salt and Pepper noise (must be in the (0,1) range):\n"))
        # Applying Salt and Pepper noise to the image
        noise_image, image = saltAndPepper(image, prob)
        break
    elif choice == 2:
        prob = float(input("Enter the probability for Salt noise (must be in the (0,1) range):\n"))
        # Applying Salt noise to the image
        noise_image, image = salt(image, prob)
        break
    elif choice == 3:
        prob = float(input("Enter the probability for Pepper noise (must be in the (0,1) range):\n"))
        # Applying Pepper noise to the image
        noise_image, image = pepper(image, prob)
        break
    else:
        print("Enter an appropriate number!\n")

# Converting the noise image to a 4D one
noise_image = np.reshape(noise_image, (1, noise_image.shape[0], noise_image.shape[1], noise_image.shape[2]))

while True:
    print("Enter 1 to filter the noise image using Running Max filtering:")
    print("Enter 2 to filter the noise image using Running min filtering:")
    choice = int(input())
    if choice == 1:
        filtered_image, noise_image = runningMaxFiltering(noise_image, filter)
        break
    elif choice == 2:
        filtered_image, noise_image = runningMinFiltering(noise_image, filter)
        break
    else:
        print("Enter an appropriate number!\n")

# Stacking the noise and filtered image horizontally so that they may be displayed side-by-side
display_window = np.hstack((noise_image, filtered_image))

# Displaying stacked images
cv2.imshow('Noise image (left) and filtered image (right)', display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
