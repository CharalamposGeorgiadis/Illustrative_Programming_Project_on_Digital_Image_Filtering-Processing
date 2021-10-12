import numpy as np
import cv2


# Function that isolates the part of the image that will be zoomed
# param image: 4D array that holds the original image
# param z: zooming factor
# param c: the user's choice for which part of the image will be zoomed
# returns: zoomed image, original image as a 3D array
def zoom_calculator(img, z, c):
    h = img.shape[0]
    w = img.shape[1]
    to_keepX = h // z
    to_keepY = w // z
    if c == 1:
        # Keeping the top-left part of the image
        to_zoom = img[:to_keepX, :to_keepY]
    elif c == 2:
        # Keeping the top-middle part of the image
        to_cutY = (w - to_keepY) // 2
        to_zoom = img[:to_keepX, to_cutY:w - to_cutY]
    elif c == 3:
        # Keeping the top-right part of the image
        to_zoom = img[:to_keepX, to_keepY:w]
    elif c == 4:
        # Keeping the middle-left part of the image
        to_cutX = (h - to_keepX) // 2
        to_zoom = img[to_cutX:h - to_cutX, :to_keepY]
    elif c == 5:
        # Keeping the middle part of the image
        to_cutX = (h - to_keepX) // 2
        to_cutY = (w - to_keepY) // 2
        to_zoom = img[to_cutX:h - to_cutX, to_cutY:w - to_cutY]
    elif c == 6:
        # Keeping the middle-right part of the image
        to_cutX = (h - to_keepX) // 2
        to_zoom = img[to_cutX:h - to_cutX, to_keepY:w]
    elif c == 7:
        # Keeping the bottom-left part of the image
        to_zoom = img[to_keepX:h, :to_keepY]
    elif c == 8:
        # Keeping the bottom-middle part of the image
        to_cutY = (w - to_keepY) // 2
        to_zoom = img[to_keepX:h, to_cutY:w - to_cutY]
    elif c == 9:
        # Keeping the bottom-right part of the image
        to_zoom = img[to_keepX:h, to_keepY:w]
    return to_zoom


# Function that takes as input an image and a zooming factor and performs image zooming
# param image: 4D array that holds the original image
# param zoom_factor: zooming factor
# param ch: the user's choice for which part of the image will be zoomed
# returns: zoomed image, original image as a 3D array
def zooming(img, zoom_factor, ch):
    h, w, c = img.shape[1], img.shape[2], img.shape[3]
    img = np.reshape(img, (h, w, c))
    output = np.zeros(img.shape)
    gif = np.array(img)
    frames = []
    to_zoom = zoom_calculator(img, zoom_factor, ch)
    # Placing each pixel to its corresponding position in the zoomed image
    for i in range(to_zoom.shape[0]):
        i_zoom = i * zoom_factor
        for j in range(to_zoom.shape[1]):
            j_zoom = j * zoom_factor
            for k in range(zoom_factor):
                for l in range(zoom_factor):
                    if k + i_zoom < output.shape[0] and l + j_zoom < output.shape[1]:
                        output[k + i_zoom][l + j_zoom] = to_zoom[i][j]
                        gif[k + i_zoom][l + j_zoom] = to_zoom[i][j]
        frames.append(gif.astype(np.uint8))
    # Displaying a gif of how the image is zoomed row-by-row
    for i in range(len(frames)):
        cv2.imshow('Image zooming row-by-row', frames[i].astype(np.uint8))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output, img


# Reading an image
image = cv2.imread('inputs/input1.jpg')

# Resizing the original image
image = cv2.resize(image, (600, 480))

# Converting the image to a 4D one
if len(image.shape) == 2:
    image = np.reshape(image, (1, 480, 600, 1))
elif len(image.shape) == 3:
    image = np.reshape(image, (1, 480, 600, 3))

# The user enter the zooming factor for image zooming
zoom = int(input('Enter the zooming factor (Must be a positive integer):\n'))

print("Choose which part of the image will be zoomed:")
while True:
    print("Enter 1 to zoom the top-left part of the image:")
    print("Enter 2 to zoom the top-middle part of the image:")
    print("Enter 3 to zoom the top-right part of the image:")
    print("Enter 4 to zoom the middle-left part of the image:")
    print("Enter 5 to zoom the middle part of the image:")
    print("Enter 6 to zoom the middle-right part of the image:")
    print("Enter 7 to zoom the bottom-left part of the image:")
    print("Enter 8 to zoom the bottom-middle part of the image:")
    print("Enter 9 to zoom the bottom-right part of the image:")
    choice = int(input())
    if choice < 1 or choice > 9:
        print("Enter an appropriate number!\n")
    else:
        # Applying Image Zooming on the image
        zoom_image, image = zooming(image, zoom, choice)
        break

# Stacking the original and zoomed image horizontally so that they may be displayed side-by-side
display_window = np.hstack((image, zoom_image))

# Displaying the original image alongside the zoomed image
cv2.imshow('Original image (left) and zoomed image (right) with a zoom factor of ' + str(zoom),
           display_window.astype(np.uint8))
# Wait for key press
cv2.waitKey(0)
# Closing image window
cv2.destroyAllWindows()
