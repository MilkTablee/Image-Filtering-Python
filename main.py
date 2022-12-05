'''
THIS ASSIGNMENT WAS DONE ON PyCharm WITH PYTHON VERSION 3.6.6 AND THE PACKAGE
opencv-contrib-python VERSION 3.4.2.16
'''
import numpy as np
import cv2


# Function to show image in a window
def cv2_imshow(image):
    cv2.imshow('Convolved Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to convert image to grayscale
def grayScale(image):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return gray


# Function that reimplements cv2.filter2D function
def filter2D(image, kernel, padding=0, strides=1):
    # Get Shapes of Kernel and Image
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Image (Convolved Image)
    xOutput = int(((xImgShape + padding - xKernShape) / strides) + 1)
    yOutput = int(((yImgShape + padding - yKernShape) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply equal Padding to all sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()/255
                except:
                    break
    return output


# Main function
def main():

    # Read image
    img = cv2.imread('./victoria.jpg')

    # Convert to grayscale
    image = grayScale(img)

    # Define a Kernel
    #kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel = np.ones((5, 5), np.float32)/25

    # Convolve and visualise output
    output = filter2D(image, kernel, padding=1)
    cv2_imshow(output)


main()
