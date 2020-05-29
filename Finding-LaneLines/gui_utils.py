  
# MIT License

# Copyright (c) 2016 Maunesh Ahir
import numpy as np
import cv2

def draw_lines(img, lines, color=(0, 0, 255), thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

class EdgeFinder:
    def __init__(self, image, filter_size=1, threshold1=0, threshold2=0):
        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)

        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('smoothed')

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        self._smoothed_img = cv2.GaussianBlur(self.image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)
        cv2.imshow('smoothed', self._smoothed_img)
        cv2.imshow('edges', self._edge_img)

class Hough:
    def __init__(self, image, rho=1, theta=1, threshold=1, min_line_length=1, max_line_gap=1):
        self.image = image
        self._rho = rho
        self._theta = theta
        self._threshold = threshold
        self._min_line_length = min_line_length
        self._max_line_gap = max_line_gap

        def onchangeRho(pos):
            self._rho = pos
            self._render()

        def onchangeTheta(pos):
            self._theta = np.pi/180*pos
            self._render()
        
        def onchangeThreshold(pos):
            self._threshold = pos
            self._render()

        def onchangeMinLine(pos):
            self._min_line_length = pos
            self._render()

        def onchangeMaxLine(pos):
            self._max_line_gap = pos
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('rho', 'edges', self._rho, 50, onchangeRho)
        cv2.createTrackbar('theta', 'edges', self._theta, 50, onchangeTheta)
        cv2.createTrackbar('threshold', 'edges', self._threshold, 50, onchangeThreshold)
        cv2.createTrackbar('min line length', 'edges', self._min_line_length, 255, onchangeMinLine)
        cv2.createTrackbar('max line gap', 'edges', self._max_line_gap, 255, onchangeMaxLine)


        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')

    def rho(self):
        return self._rho

    def theta(self):
        return self._theta

    def threshold(self):
        return self._threshold

    def min_line_length(self):
        return self._min_line_length

    def max_line_gap(self):
        return self._max_line_gap
    
    def houghImage(self):
        return self._hough_img

    def _render(self):
        self._hough_img = hough_lines(self.image, self._rho, self._theta, self._threshold, self._min_line_length, self._max_line_gap)
        cv2.imshow('hoguh', self._hough_img)