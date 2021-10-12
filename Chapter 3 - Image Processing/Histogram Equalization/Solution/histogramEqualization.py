from matplotlib import pyplot as plt
import cv2
import numpy as np


# Function that calculates the histogram of an image
# param img: 4D array containing an image
# return: Histogram of the image, original image as a 3D array
def histogramCalculation(img):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c)).astype(np.uint8)
    output = np.zeros(256)
    flat_image = img.flatten()
    for i in range(len(flat_image)):
        output[flat_image[i]] += 1
    return output, img


# Function that calculates the cumulative sum of a histogram
# param h: histogram
# returns: cdf of the histogram
def cumulative_sum(h):
    sum = np.zeros(h.shape)
    for i in range(len(h)):
        for k in range(i + 1):
            sum[i] += h[k]
    return sum


# Function that performs Histogram Equalization on an image
# param img: 4D array containing an image
# param cd: cdf of the image
# return: Equalized image, original image as a 3D array
def histogramEqualization(img, cd):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c)).astype(np.uint8)
    # Performing Histogram Equalization
    cdf_mask = np.ma.masked_equal(cd, 0)
    cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
    eq_img = cdf_mask[img]
    return eq_img, img


# Function that plot the histogram and cdf of an image
# param h: Histogram
# param eq: Boolean containing whether the histogram has been equalized or not
# param c: cdf of the histogram
def plot_histogram(h, c, eq):
    plt.plot(c, color='b')
    plt.plot(h, color='r')
    plt.xlim([0, 256])
    if eq:
        plt.legend(('CDF', 'Equalized Histogram'), loc='upper left')
    else:
        plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.xlabel("Intensity Level")
    plt.ylabel("Intensity Frequency")
    plt.show()


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (600, 480), 0)

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 480, 600, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 480, 600, 3))

# Calculating the histogram of the image
hist, image = histogramCalculation(image)
# Calculating the cdf of the histogram
cdf = cumulative_sum(hist)
# Normalizing the cdf
cdf = cdf * np.amax(hist) / np.amax(cdf)

# Plotting the cdf and histogram of the image
plot_histogram(hist, cdf, False)

# Converting the image to a 4D one
image = np.reshape(image, (1, 480, 600, image.shape[2]))

# Performing Histogram Equalization to the image
eq_image, image = histogramEqualization(image, cdf)

# Converting the equalized image to a 4D one
eq_image = np.reshape(eq_image, (1, 480, 600, eq_image.shape[2]))

# Calculating the histogram of the image
eq_hist, eq_image = histogramCalculation(eq_image)
# Calculating the cdf of the histogram
eq_cdf = cumulative_sum(eq_hist)
# Normalizing the cdf
eq_cdf = eq_cdf * np.amax(eq_hist) / np.amax(eq_cdf)

# Plotting the cdf and histogram of the equalized image
plot_histogram(eq_hist, eq_cdf, True)

# Stacking the original and equalized image horizontally so that they may be displayed side-by-side
display_window = np.hstack((image, eq_image))

# Displaying stacked images
cv2.imshow("Original image (left) and Equalized image (right)", display_window.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
