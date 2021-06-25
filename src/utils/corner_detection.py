import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../../1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, .04)
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
image[dst > 0.05 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
