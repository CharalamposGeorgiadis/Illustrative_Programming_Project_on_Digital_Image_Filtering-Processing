import numpy as np
import cv2


# Function that performs Image Halftoning using Pseudorandom Thresholding
# param img: 4D array containing a grayscale image
# param dither_size: size of dither matrix
# returns: halftoned image, input image as a 3D array
def pseudorandomThersholding(img, dither_size):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    frames = []
    gif = np.array(img)
    # Initializing the dither matrix
    dither_matrix = np.zeros((dither_size, dither_size), np.int)
    cs = 2
    for i in range(int(np.log(dither_size) / np.log(2))):
        for j in range(cs // 2):
            for k in range(cs // 2):
                dither_matrix[j][k] *= 4
                dither_matrix[j + (cs // 2)][k] = dither_matrix[j][k] + 3
                dither_matrix[j][k + (cs // 2)] = dither_matrix[j][k] + 2
                dither_matrix[j + (cs // 2)][k + (cs // 2)] = dither_matrix[j][k] + 1
        cs *= 2
    # Scaling dither matrix elements
    dmax, dmin = 0, 0
    for i in range(dither_size):
        for j in range(dither_size):
            if dither_matrix[i][j] > dmax:
                dmax = dither_matrix[i][j]
            if dither_matrix[i][j] < dmin:
                dmin = dither_matrix[i][j]
    # Float that determines the scaling of the dither matrix elements
    scale = 255 / (dmax - dmin)
    for i in range(dither_size):
        for j in range(dither_size):
            # dmin is usually 0
            dither_matrix[i][j] = dmin + scale * (dither_matrix[i][j] - dmin)
    # Applying dithering
    for i in range(h):
        for j in range(w):
            # The mod operation ensures that every threshold in the dither matrix is used in an orderly manner
            if img[i][j] > dither_matrix[i % dither_size][j % dither_size]:
                output[i][j] = 255
            else:
                output[i][j] = 0
            gif[i][j] = output[i][j]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how pseudorandom thresholding is applied to the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Pseudorandom thresholding row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading an image as grayscale
image = cv2.imread('Inputs/input1.jpg', 0)

# Resizing the original image
image = cv2.resize(image, (640, 480))

image = np.reshape(image, (1, 480, 640, 1))

# The user enters the the size of the dither matrix
size = int(input('Enter the size of the dither matrix (2 creates a 2x2 matrix). It must be a power of 2:\n'))

# Applying image halftoning using pseudorandom thresholding
pseudo_image, image = pseudorandomThersholding(image, size)

# Stacking the images horizontally so that they be displayed side-by-side
displayWindow = np.hstack((image, pseudo_image))

# Displaying the salt images alongside the filtered ones
cv2.imshow('Original image (left) and halftoned image using Pseudorandom Thresholding (right)',
           displayWindow.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
