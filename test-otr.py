#!/usr/bin/env python3
import cv2
import numpy as np
import TableRecognition


def runOTR(filename):
    img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File {0} does not exist".format(filename))
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGrey, 150, 255, cv2.THRESH_BINARY_INV)[1]
    imgThreshInv = cv2.threshold(imgGrey, 150, 255, cv2.THRESH_BINARY)[1]

    imgDil = cv2.dilate(imgThresh, np.ones((5, 5), np.uint8))
    imgEro = cv2.erode(imgDil, np.ones((4, 4), np.uint8))

    contour_analyzer = TableRecognition.ContourAnalyzer(imgDil)
    # 1st pass (black in algorithm diagram)
    contour_analyzer.filter_contours(min_area=400)
    contour_analyzer.build_graph()
    contour_analyzer.remove_non_table_nodes()
    contour_analyzer.compute_contour_bounding_boxes()
    contour_analyzer.separate_supernode()
    contour_analyzer.find_empty_cells(imgThreshInv)

    contour_analyzer.find_corner_clusters()
    contour_analyzer.compute_cell_hulls()
    contour_analyzer.find_fine_table_corners()

    # Add missing contours to contour list
    missing_contours = contour_analyzer.compute_filtered_missing_cell_contours()
    contour_analyzer.contours += missing_contours

    # 2nd pass (red in algorithm diagram)
    contour_analyzer.compute_contour_bounding_boxes()
    contour_analyzer.find_empty_cells(imgThreshInv)

    contour_analyzer.find_corner_clusters()
    contour_analyzer.compute_cell_hulls()
    contour_analyzer.find_fine_table_corners()

    # End of 2nd pass. Continue regularly
    contour_analyzer.compute_table_coordinates(5.)

    contour_analyzer.draw_table_coord_cell_hulls(img, xscale=.8, yscale=.8)
    return img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The image file to read')
    parser.add_argument('-o', '--outfile', default="out.png", help='The output file.')
    args = parser.parse_args()

    img = runOTR(args.infile)

    cv2.imwrite(args.outfile, img)
