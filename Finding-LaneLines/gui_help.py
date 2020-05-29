"""
How to run:
python find_edges.py <image path>
original author was # MIT License # Copyright (c) 2016 Maunesh Ahir

I adapted poorly to operate on Hough transforms
"""

import argparse
import cv2
import os
import numpy as np
from process_script import grayscale, gaussian_blur, canny, region_of_interest

from gui_utils import EdgeFinder, Hough


def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')
    args = parser.parse_args()
    img = cv2.imread(args.filename)
    cv2.imshow('input', img)

    
    # Create grayscale of img
    gray = grayscale(img)

    # visualize the canny edge detection
    edge_finder = EdgeFinder(gray)

    (head, tail) = os.path.split(args.filename)
    (root, ext) = os.path.splitext(tail)
    edge_filename = os.path.join("output_images", root + "-edged" + ext)
    cv2.imwrite(edge_filename, edge_finder.edgeImage())

    # masking
    ysize = img.shape[0]
    xsize = img.shape[1]
    left_flat = [100, ysize]
    apex = [xsize/2, ysize/2+20]
    right_flat = [xsize-50, ysize]

    points = [left_flat, apex, right_flat]
    vertices = np.array([points], dtype=np.int32)

    masked_edges = region_of_interest(edge_finder.edgeImage(), [vertices])    

    # call hough class
    hough_finder = Hough(masked_edges)

    print("Hough parameters:")
    print("Rho: %f" % hough_finder.rho())
    print("Theta: %f" % hough_finder.theta())
    print("Threshold: %f" % hough_finder.threshold())
    print("Min Line Length: %f" % hough_finder.min_line_length())
    print("Max Line Gap: %f" % hough_finder.max_line_gap())

    (head, tail) = os.path.split(args.filename)

    (root, ext) = os.path.splitext(tail)

    hough_filename = os.path.join("output_images", root + "-houghed" + ext)

    cv2.imwrite(hough_filename, hough_finder.houghImage())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()