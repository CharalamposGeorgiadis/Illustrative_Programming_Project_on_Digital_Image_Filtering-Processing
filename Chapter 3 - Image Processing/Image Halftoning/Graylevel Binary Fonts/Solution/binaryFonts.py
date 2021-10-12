import numpy as np
import cv2


# Function that performs Image Halftoning using Graylevel Binary Fonts
# param img: 4D array containing a grayscale image
# param h: number of halftones
# returns: halftoned image, input image as a 3D array
def gray_levelHalftoning(img, halftones):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    n = int(0.001 + np.sqrt(halftones))
    halftones = 256 / halftones
    frames = []
    gif = np.array(img)
    for i in range(0, h - n + 1, n):
        for j in range(0, w - n + 1, n):
            count = img[i][j] // halftones
            for k in range(i, i + n):
                for l in range(j, j + n):
                    if count != 0:
                        output[k][l] = 255
                        gif[k][l] = output[k][l]
                        count -= 1
                    else:
                        break
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how Gray-level halftoning is applied to the image row-by-row
    for i in range(len(frames)):
        cv2.imshow('Gray-level halftoning row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading an image as grayscale
image = cv2.imread('Inputs/input1.jpg', 0)

# Resizing the original image
image = cv2.resize(image, (640, 480))

image = np.reshape(image, (1, 480, 640, 1))

# The user enters the number of halftones
number_of_halftones = int(input('Enter the number of halftones (Must be a positive integer):\n'))

# Applying image halftoning using graylevel binary fonts
gray_level_image, image = gray_levelHalftoning(image, number_of_halftones)

# Stacking the images horizontally so that they be displayed side-by-side
displayWindow = np.hstack((image, gray_level_image))

# Displaying the salt images alongside the filtered ones
cv2.imshow('Original image (left) and halftoned image using Graylevel Binary Fonts (right)',
           displayWindow.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
