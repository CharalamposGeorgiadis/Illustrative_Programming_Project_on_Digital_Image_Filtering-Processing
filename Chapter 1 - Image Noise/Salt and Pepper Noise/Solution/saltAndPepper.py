import numpy as np
import cv2


# Function that adds salt and pepper noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def saltAndPepper(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    gif = np.array(img)
    frames = []
    for i in range(h):
        for j in range(w):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is between 0 and p / 2, then the current pixel's value will be set to 0 (black)
            if rand < p / 2:
                output[i][j] = 0
            # If the random number is between p / 2 and p, then the current pixel's value will be set to 255 (white)
            elif rand < p:
                output[i][j] = 255
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
            gif[i][j] = output[i][j]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the image is corrupted by Salt and Pepper noise row-by-row
    for i in range(len(frames)):
        cv2.imshow('Salt and pepper noise corruption row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Function that adds salt noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def salt(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    gif = np.array(img)
    frames = []
    for i in range(h):
        for j in range(w):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is more than 1 - p, then the current pixel's value will be set to 255 (white)
            if rand > 1 - p:
                output[i][j] = 255
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
            gif[i][j] = output[i][j]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the image is corrupted by Salt noise row-by-row
    for i in range(len(frames)):
        cv2.imshow('Salt noise corruption row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Function that adds pepper noise to an image
# param img: 4D array containing the original, clean image
# param p : noise probability
# returns: noise image, original image as a 3D array
def pepper(img, p):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape, np.uint8)
    gif = np.array(img)
    frames = []
    for i in range(h):
        for j in range(w):
            # Creating a random float between 0 and 1 for every pixel of the image
            rand = np.random.rand()
            # If the random number is less than the probability, then the current pixel's value will be set to 0 (black)
            if rand < p:
                output[i][j] = 0
            # Otherwise, the pixel's value remains unchanged
            else:
                output[i][j] = img[i][j]
            gif[i][j] = output[i][j]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the image is corrupted by Pepper noise row-by-row
    for i in range(len(frames)):
        cv2.imshow('Pepper noise corruption row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading the image
image = cv2.imread('Inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (450, 360))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 360, 450, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 360, 450, 3))

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

# Stacking the clean and noise image horizontally so that they may be displayed side-by-side
display_window = np.hstack((image, noise_image))
# Displaying stacked images
cv2.imshow('Original image (left) and noise image (right)', display_window)
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
