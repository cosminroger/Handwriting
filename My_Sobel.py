from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy.signal import find_peaks
import numpy as np

def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)  



img = rgb2gray(imread('P632-Fg002-R-C01-R01-binarized.jpg'))    #Read image
height, width = img.shape   #Get size of image

sobel_image = sobel(img)   #Apply Sobel edge detection filter on image  ///// For later


hpp = horizontal_projections(sobel_image)    #Find horizontal projection of pixels per row in image
plt.plot(hpp)
plt.show()


peaks, _ = find_peaks(hpp, distance=150, height = 10)    #Find local maxima peaks in plot
plt.plot(hpp)
plt.plot(peaks, hpp[peaks], "x")
plt.show()

for x in range(len(peaks)):                     #Plot peaks on image
	plt.axline((0,peaks[x]),(width,peaks[x]))



plt.imshow(img, cmap="gray")
plt.show()

