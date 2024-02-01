# Panoramic Image Stitching

This project is a Python application that stitches together a set of images to create a panoramic view. It uses OpenCV and feature detection methods such as SIFT and ORB.

## Requirements

- Python 3.6+
- OpenCV
- numpy

## Usage

You can run the script `create_panoramic.py` with the following command:

```bash
python create_panoramic.py [-h] [-o OUTPUT] [-t THRESHOLD] [-m {SIFT,ORB}] [-v] [-c] -i IMAGE_PATHS [IMAGE_PATHS ...]

```

Arguments
-i or --image_paths: Paths to the 3 images to be stitched in order left-to-right.
-o or --output: Path to the output directory. Default is panoramic_result.png.
-t or --threshold: Threshold for the ratio test. Default is 0.6.
-m or --method: Method for feature detection. Choices are SIFT and ORB. Default is SIFT.
-v or --verbose: Print verbose output. This is optional.
-c or --crop: Crop the result image. This is optional.
License
This project is licensed under the terms of the MIT license.