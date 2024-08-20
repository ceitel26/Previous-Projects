import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


arr = cv2.imread("anko_3-4th.jpg")
print(type(arr))

# -- An error will occur here since I am not going to upload any images.
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

countDict = {}
(h, w) = arr.shape

for y in range(h):
    for x in range(w):
        colorVal = arr[y, x]
        assert (0 <= colorVal <= 255)
        countDict[colorVal] = countDict.get(colorVal, 0) + 1

values = []
for c in range(256):
    colorCount = countDict.get(c, 0)
    values.append(colorCount)

plt.bar(range(len(values)), values)
plt.show

# Load the images into lists
# Assumes you have two folders under where you run the program:
# mImages (with the images you took of your object)
allMyImages = []  # images of your desired object
allImageFiles = []  # "background" images
for fname in os.listdir("myImages"):
    path = "." + os.sep + "myImages" + os.sep + fname
    allImageFiles.append(path)
    allMyImages.append(path)

for fname in os.listdir("imagedata"):
    path = "." + os.sep + "imagedata" + os.sep + fname
    allImageFiles.append(path)

# These lists holds the SIFT descriptors and the histograms.
allSiftDesc = []
allHists = []

# Pre-compute the histograms and descriptors (this will be slow)
#  It just looks at 10 images for testing, but you can make it look at more
# or less
for path in allImageFiles[:10]:
    arr = cv2.imread(path)
    print(path)
    # TODO: compute histograms using computeHistNormalized() and SIFT descriptors
    # using OpenCV and append them to allSiftDesc and allHists

# Designate one image as the "target" ( I just picked the first of the images in a set,
# but you can change this)
targetPath = allMyImages[0]

arr = cv2.imread(targetPath)

# Build up lists of tuples,  (imagePath, distance)
histDistTuples = []  # Histogram Distances
siftDistTuples = []


#-- Sorts the list by second element of distance tuple, indexing on tuples can be touchy
# but you cannot unpack in a lamda function
sortedHistTuples = sorted(histDistTuples, key=lambda x: x[1])
sortedSIFTTuples = sorted(siftDistTuples, key=lambda x: x[1])
# Show the original image and the top 5 matches
cv2.imshow("anko_3-4th.jpg", arr)
for i in range(5):
    cv2.imshow(str(i) + "_Hist", cv2.imread(sortedHistTuples[i][0]))
    cv2.imshow(str(i) + "_SIFT", cv2.imread(sortedSIFTTuples[i][0]))
cv2.waitKey()
cv2.destroyAllWindows()


