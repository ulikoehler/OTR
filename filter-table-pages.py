#!/usr/bin/env python3
"""
Utility script that separated images that have a table from those who have none.

This script creates two directories, <indir>.tables and <indir>.notables
where both classes of images will be symlinked.
"""
import cv2
import os
import os.path
import concurrent.futures
import numpy as np
import TableRecognition
from ansicolor import red, green


def hasTable(filename, min_fract_area=.2, min_cells=150):
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

    return contour_analyzer.does_page_have_valid_table(min_fract_area, min_cells)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='The directory containing image files')
    args = parser.parse_args()

    # Build list of files to check
    tocheck = []
    for child in os.listdir(args.directory):
        fullpath = os.path.join(args.directory, child)
        if os.path.isfile(fullpath):
            tocheck.append(fullpath)

    # Perform parallel checking
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(hasTable, tocheck)

    # Create output directories (with symlinks)
    dirname = args.directory.strip("/")
    tabledir = dirname + ".table"
    notabledir = dirname + ".notable"

    if not os.path.isdir(tabledir):
        os.mkdir(tabledir)
    if not os.path.isdir(notabledir):
        os.mkdir(notabledir)

    # Iterate over results
    for filename, result in zip(tocheck, results):
        basename = os.path.basename(filename)
        if result:  # Table found
            print(green("Found table in {0}".format(filename), bold=True))
            # Create symlink
            os.symlink(os.path.join("..", filename), os.path.join(tabledir, basename))
        else:  # No table found
            print(red("Did not find table in {0}".format(filename), bold=True))
            # Create symlink
            os.symlink(os.path.join("..", filename), os.path.join(notabledir, basename))

