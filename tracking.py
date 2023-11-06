"""
This script contains function that plots the ground-truth and spatialized trajectory.
"""
import numpy as np
import cv2

def get_track(path_to_images, N_frames):
    for i in range(N_frames):
        img = cv2.imread('{}/{}.jpg'.format(path_to_images, i))
        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 3)
        canny = cv2.Canny(median, 100, 200)
        points = np.argwhere(canny > 0)
        center, radius = cv2.minEnclosingCircle(points)
        print('center:', center, 'radius:', radius)

        # draw circle on copy of input
        result = img.copy()
        x = int(center[1])
        y = int(center[0])
        rad = int(radius)
        cv2.circle(result, (x, y), rad, (255, 255, 255), 1)

        # write results
        # cv2.imwrite("sunset_canny.jpg", canny)
        # cv2.imwrite("sunset_circle.jpg", result)

        # show results
        cv2.imshow("median", median)
        cv2.imshow("canny", canny)
        cv2.imshow("result", result)
        cv2.waitKey(0)



if __name__ == '__main__':
    PATH_TO_IMAGE = './viz_output'
    get_track(PATH_TO_IMAGE, 2)