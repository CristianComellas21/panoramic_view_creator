import cv2 as cv
import numpy as np


# def get_images():
#     """
#     This function is used to get the images from the user
#     :return: list of images
#     """
#     images = []
#     while True:
#         image = input("Enter the image path: ")
#         if image == "done":
#             break
#         images.append(image)
#     return images


def create_panoramic_view(
        images: [np.ndarray],
        match_threshold: float = 0.75,

        ) -> np.ndarray:
    """
    This function is used to create the panoramic view from a list of images.

    Parameters:
    -----------
    :type images: list

    Returns:
    --------
    :rtype: numpy.ndarray
    """

    # Get the descriptors and key points of the images
    descriptors, key_points = __get_descriptors_and_key_points(images)
    
    # Create the matcher object
    matcher = cv.BFMatcher()

    # Get the matches between consecutive images
    matches = []
    for i in range(len(images) - 1):
        matches.append(matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2))

    # Get the good matches
    good_matches = []
    for match in matches:
        good_match = []
        for m, n in match:

            # Check if the distance between the descriptors is less than the threshold
            if m.distance < match_threshold * n.distance:
                good_match.append(m)
        good_matches.append(good_match)



    
### Private Functions ###

def __get_descriptors_and_key_points(
        images: [np.ndarray],
        ) -> ([np.ndarray], [np.ndarray]):
    """
    This function is used to get the descriptors and key points of the images.

    Parameters:
    -----------
    :type images: list

    Returns:
    --------
    :rtype: tuple
    """
    # Create the SIFT object
    sift = cv.SIFT_create()

    # Get the descriptors and key points of the images
    key_points = []
    descriptors = []
    for image in images:
        key_point, descriptor = sift.detectAndCompute(image, None)
        key_points.append(key_point)
        descriptors.append(descriptor)

    return descriptors, key_points

