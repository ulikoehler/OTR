#!/usr/bin/env python3
"""
Fetch oldweather images for OpenCV tests
"""
import subprocess
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('year')
    args = parser.parse_args()
    name = args.name

    if name == "Northwind":  # 1946 - 1948
        url = "https://zooniverse-static.s3.amazonaws.com/old-weather-2015/Cold_Science/Coast_Guard/Northwind_WAG-282_or_WAGB-282_/Northwind-WAG-282-{0}-split/Northwind-WAG-282-{0}-{1:04d}-{2}.JPG"
    elif name == "Northland":  # 1928 - 1930
        url = "https://zooniverse-static.s3.amazonaws.com/old-weather-2015/The_Arctic_Frontier/Coast_Guard/Northland_WPG-49_/Northland-WPG-49-{0}-split/Northland-WPG-49-{0}-{1:04d}-{2}.JPG"
    elif name == "Eastwind": # 1946 - 1948
        url = "https://zooniverse-static.s3.amazonaws.com/old-weather-2015/Cold_Science/Coast_Guard/Eastwind_WAG-279_or_WAGB-279_/eastwind-wag-279-{0}-split/eastwind-wag-279-{0}-{1:04d}-{2}.JPG"
    else:
        raise ValueError("Invalid name: " + name)
    urls = [url.format(args.year, i, 0) for i in range(1000)]
    urls += [url.format(args.year, i, 1) for i in range(1000)]

    # Create directory if it does not exist
    if not os.path.isdir(name):
        os.mkdir(name)
    # Use wget for parallel downloading
    subprocess.call(["wget", "-P", name, "-c"] + urls)