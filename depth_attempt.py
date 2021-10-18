import cv2
import numpy as np
from matplotlib import pylab as plt

if __name__ == "__main__":
    left_name  = "C:\\Users\\Maxli\\Pictures\\Camera Roll\\WIN_20211017_22_19_49_Pro.jpg"
    right_name = "C:\\Users\\Maxli\\Pictures\\Camera Roll\\WIN_20211017_22_19_53_Pro.jpg"

    left  = cv2.imread(left_name,0)
    right = cv2.imread(right_name,0)

    cv2.imshow("window", left)
    cv2.waitKey(0)
    cv2.imshow("window", right)
    cv2.waitKey(0)

    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    disparity = stereo.compute(left, right)
    #plt.figure(figsize=(20, 10))
    #plt.imshow(disparity, 'disparity')
    #plt.xticks([])
    #plt.yticks([])
    cv2.imshow("window", disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
