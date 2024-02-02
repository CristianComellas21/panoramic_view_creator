"""
Executable file for creating panoramic view from a set of images.
"""

from argparse import ArgumentParser
from algorithm import create_panoramic_view
import numpy as np
import cv2 as cv
from pathlib import Path


def main():

    # === ARGUMENT PARSING ===

    # Define the argument parser
    arg_parser = ArgumentParser()

    # --- Add named arguments
    arg_parser.add_argument("-o", "--output", default="output/panoramic_result.png", help="Path to the output directory.")

    arg_parser.add_argument("-r", "--ratio", default=0.65, type=float, help="Ratio for the correct matches selection.")
    
    arg_parser.add_argument("-m", "--method", choices=["SIFT", "ORB"], default="SIFT", help="Method for feature detection.")

    # --- Add action arguments
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    
    arg_parser.add_argument("-c", "--crop", action="store_true", help="Crop the result image.")

    # --- Add positional arguments (image paths)
    arg_parser.add_argument("-i", "--image_paths", required=True, nargs="+", type=str, help="Path to the 3 images to be stitched in order left-to-right.")


    # Parse the arguments
    args = arg_parser.parse_args()

    # Extract the arguments
    output_path = args.output
    match_ratio = args.ratio
    image_paths = args.image_paths
    crop = args.crop
    method = args.method
    verbose = args.verbose

    # ------------------------

    # === EXECUTION ===

    # Read the images
    images = [__read_image(image_path) for image_path in image_paths]

    # Create the panoramic view
    panoramic = create_panoramic_view(
        images=images, 
        match_ratio=match_ratio,
        crop=crop,
        detector=method,
        verbose=verbose)

    # Save the result
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(output_path, panoramic)

    # -----------------


def __read_image(image_path: str) -> np.ndarray:
    """
    This function is used to read an image from a given path.

    Parameters:
    -----------
    :type image_path: pathlib.Path

    Returns:
    --------
    :rtype: numpy.ndarray
    """
    image = cv.imread(image_path)
    return image


if __name__ == "__main__":
    main()