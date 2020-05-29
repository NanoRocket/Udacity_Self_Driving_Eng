import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(0, 0, 255), thickness=4):
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
    left_x_points = []
    left_y_points = []
    right_x_points = []
    right_y_points = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            if slope < 0: 
                left_x_points.append(x1)
                left_x_points.append(x2)
                left_y_points.append(y1)
                left_y_points.append(y2)

            else:
                right_x_points.append(x1)
                right_x_points.append(x2)
                right_y_points.append(y1)
                right_y_points.append(y2)

    # SHOULD MAKE A NEW FUNCTION FOR THIS
    # want to extrapolate up to the highest y value
    max_y_coord = min(min(left_y_points), min(right_y_points))

    left_slope, left_intercept = np.polyfit(left_x_points, left_y_points,1)
    right_slope, right_intercept = np.polyfit(right_x_points, right_y_points, 1)

    # get our 8 points to plot:
    bottom = img.shape[0]
    
    ly1 = bottom
    #ly2 = max_y_coord
    ly2 = bottom//2 + 60
    lx1 = int((ly1-left_intercept)/left_slope)
    lx2 = int((ly2-left_intercept)/left_slope)
    
    ry1 = bottom
    #ry2 = max_y_coord
    ry2 = bottom//2 + 60
    rx1 = int((ry1-right_intercept)/right_slope)
    rx2 = int((ry2-right_intercept)/right_slope)

    cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)
    cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, γ)

import os

directory = 'test_images'
directory_save = 'test_images_output/'

# The following comes primarily from the Udacity quiz, only parameters changed.
for image_name in os.listdir(directory):
    # Read in and grayscale the image
    image = cv2.imread('test_images/' + str(image_name))
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 17
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 10
    high_threshold = 40
    edges = canny(blur_gray, low_threshold, high_threshold)

    # masking
    left_flat = [100, 540]
    apex = [480, 290]
    right_flat = [910, 540]

    points = [left_flat, apex, right_flat]
    vertices = np.array([points], dtype=np.int32)

    masked_edges = region_of_interest(edges, [vertices])
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5  #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    hough_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
 
    # Draw the lines on the edge image
    # Create a "color" binary image to combine with line image
    line_copy= np.copy(image) # creating a blank to draw lines on

    lines_edges = weighted_img(hough_img, line_copy, .8, 1.0, 0.0) 
    plt.imshow(lines_edges)
    cv2.imwrite(os.path.join(directory_save, str(image_name)), lines_edges)
