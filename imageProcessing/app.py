import cv2
import numpy as np

# dummy function that does nothing
def dummy(value):
    pass

# define convolution kernels
identityKernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussianKernel = cv2.getGaussianKernel(3, 0)
gaussianKernel2 = cv2.getGaussianKernel(5, 0)
boxKernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9.0

kernels = [identityKernel, sharpenKernel, gaussianKernel, gaussianKernel2, boxKernel]


# Read in an image, make a grayscale copy
imagePath = 'imageProcessing/kitten.jpg'

colorOriginal = cv2.imread(imagePath)
grayOriginal = cv2.cvtColor(colorOriginal, cv2.COLOR_BGR2GRAY)


# Creae the UI (window and trackbars)
cv2.namedWindow('app')


# arguments: trackbarName, windowName, value (initial value), count (max value), onChange (event handler)
cv2.createTrackbar('contrast', 'app', 1, 100, dummy)    
cv2.createTrackbar('brightness', 'app', 50, 100, dummy)
cv2.createTrackbar('filter', 'app', 0, len(kernels)-1, dummy)   
cv2.createTrackbar('grayscale', 'app', 0, 1, dummy)

count = 0

# main UI loop
while True:
    # read all of the trackbar values
    grayScale = cv2.getTrackbarPos('grayscale', 'app')
    contrast = cv2.getTrackbarPos('contrast', 'app')
    brightness = cv2.getTrackbarPos('brightness', 'app')
    kernelIdx = cv2.getTrackbarPos('filter', 'app')

    # apply the filters
    colorModified = cv2.filter2D(colorOriginal,-1, kernels[kernelIdx])
    grayModified = cv2.filter2D(grayOriginal,-1, kernels[kernelIdx])


    #  apply the brightness and contrast
    colorModified = cv2.addWeighted(colorModified, contrast, np.zeros_like(colorOriginal), 0, brightness - 50)
    grayModified = cv2.addWeighted(grayModified, contrast, np.zeros_like(grayOriginal), 0, brightness - 50)



    # wait for keypress (100 milliseconds)
    key = cv2.waitKey(100)
    if key == ord('q'):  
        break  
    elif key == ord('s'):
        # Save image 
        if grayScale == 0:
            cv2.imwrite('output-{}.png'.format(count), colorModified)
        else:
            cv2.imwrite('output-{}.png'.format(count), grayModified)
        
        count += 1

    # show the image 
    if grayScale == 0:
        cv2.imshow('app', colorModified)
    else:
        cv2.imshow('app', grayModified)


# TODO: remove this line
cv2.waitKey(0)

# Window cleanup
cv2.destroyAllWindows()