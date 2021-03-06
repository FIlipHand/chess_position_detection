import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.cvtColor(cv2.imread("../../2.png"), cv2.COLOR_BGR2RGB)


def resize(img, height=800):
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))


img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)
img = cv2.bilateralFilter(img, 9, 75, 75)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
img = cv2.medianBlur(img, 11)
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
edges = cv2.Canny(img, 200, 250)

# Getting contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Finding contour of biggest rectangle
# Otherwise return corners of original image
# Don't forget on our 5px border!
height = edges.shape[0]
width = edges.shape[1]
MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

# Page fill at least half of image, then saving max area found
maxAreaFound = MAX_COUNTOUR_AREA * 0.5

# Saving page contour
pageContour = np.array([[5, 5], [5, height - 5], [width - 5, height - 5], [width - 5, 5]])

# Go through all contours
for cnt in contours:
    # Simplify contour
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound
    if (len(approx) == 4 and
            cv2.isContourConvex(approx) and
            maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
        maxAreaFound = cv2.contourArea(approx)
        pageContour = approx

print(pageContour)
cv2.imshow('dupa', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
