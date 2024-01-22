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
        match_threshold: float = 0.6,

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

    # # Get the descriptors and key points of the images
    # descriptors, key_points = __get_descriptors_and_key_points(images)

    # # Get the matches between consecutive images
    # # It has len(images) - 1 elements (one less than the number of images
    # matches = __get_matches(descriptors)

    # # Get the good matches
    # good_matches = __filter_matches(matches, match_threshold)

    # # Get the homography matrix for each pair of consecutive images
    # homography_matrices = get_homography_matrix(good_matches, key_points)

    # # Get the dimensions of the final image
    # height = images[0].shape[0]
    # total_width = np.sum([image.shape[1] for image in images])
    

    assert len(images) > 1, "At least two images are required to create a panoramic view."
    
    left_image = images[0]
    for idx in range(len(images)-1):
        right_image = images[idx+1]

        # Get the descriptors and key points of the images
        result = __create_panoramic_view(left_image, right_image, match_threshold=match_threshold)

        # Change the left image to the result
        left_image = result
    
    return result

    # return final_image, {
    #     "descriptors": descriptors,
    #     "key_points": key_points,
    #     "matches": matches,
    #     "good_matches": good_matches,
    #     "homography_matrices": homography_matrices,
    #     "final_homography_matrix": final_homography_matrix
    # }

    
### Private Functions ###

def __create_panoramic_view(
        image_left: np.ndarray,
        image_right: np.ndarray,
        match_threshold: float
) -> np.ndarray:
    """
    This function is used to create the panoramic view from two images.
    
    Parameters:
    -----------
    :type image_left: numpy.ndarray
    :type image_right: numpy.ndarray

    Returns:
    --------
    :rtype: numpy.ndarray
    """

    # Get the descriptors and key points of the images
    descriptors, key_points = __get_descriptors_and_key_points(image_left, image_right)

    # Get the matches between consecutive images
    matches = __get_matches(descriptors[0], descriptors[1])

    # Get the good matches
    good_matches = __filter_matches(matches, match_threshold)

    # Get the homography matrix for each pair of consecutive images
    homography_matrix = __get_homography_matrix(good_matches, key_points[0], key_points[1])

    # Warp the images
    result = __warp_images(image_right, image_left, homography_matrix)

    return result


def __get_descriptors_and_key_points(
    image_left: np.ndarray,
    image_right: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    This function is used to get the descriptors and key points of the images.

    Parameters:
    -----------
    :type images: list

    Returns:
    --------
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    # Create the ORB object
    detector = cv.ORB_create(nfeatures=2000)

    # Get the descriptors and key points of the images
    key_point_left, descriptors_left = detector.detectAndCompute(image_left, None)
    key_point_right, descriptors_right = detector.detectAndCompute(image_right, None)


    print("Descriptors Left: ", descriptors_left.shape)
    print("Descriptors Right: ", descriptors_right.shape)

    return (descriptors_left, descriptors_right), (key_point_left, key_point_right)

def __get_matches(
        descriptors_left: np.ndarray,
        descriptors_right: np.ndarray,
        ) -> [cv.DMatch]:
    """
    This function is used to get the matches between consecutive images.
    
    Parameters:
    -----------
    :type descriptor_left: numpy.ndarray
    :type descriptor_right: numpy.ndarray

    Returns:
    --------
    :rtype: list of cv.DMatch
    """

    # Create the matcher object
    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)

    # Get the matches between consecutive images
    matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

    print("Matches: ", len(matches))

    return matches


def __filter_matches(
        matches: [cv.DMatch],
        ratio: float,
        ) -> [cv.DMatch]:
    """
    This function is used to filter the matches.

    Parameters:
    -----------
    :type matches: list
    :type ratio: float

    Returns:
    --------
    :rtype: list
    """
    filtered_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * ratio:
            filtered_matches.append(m1)



    print("Good Matches: ", len(filtered_matches))

    return filtered_matches

def __get_homography_matrix(
        good_matches: [cv.DMatch],
        key_points_left: [cv.KeyPoint],
        key_points_right: [cv.KeyPoint],
        ) -> np.ndarray:
    """
    This function is used to get the homography matrix for each pair of consecutive images.

    Parameters:
    -----------
    :type good_matches: list
    :type key_points: list

    Returns:
    --------
    :rtype: list
    """
    # Get the coordinates of the matches
    points_left = np.float32([key_points_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_right = np.float32([key_points_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Get the homography matrix
    homography_matrix, _ = cv.findHomography(points_left, points_right)
    # homography_matrix, _ = cv.findHomography(points_left, points_right, cv.RANSAC, 5.0)
    return homography_matrix


def __warp_images(
        image1: np.ndarray,
        image2: np.ndarray,
        homography_matrix: np.ndarray,
        ) -> np.ndarray:
    """
    This function is used to warp the images.
    
    Parameters:
    -----------
    :type image1: numpy.ndarray
    :type image2: numpy.ndarray
    :type homography_matrix: numpy.ndarray
    
    Returns:
    --------
    :rtype: numpy.ndarray
    """

    rows1, cols1 = image1.shape[:2]
    rows2, cols2 = image2.shape[:2]

    # Get the corners of the left image
    corners1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)

    # Get the corners of the right image
    corners2_tmp = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # Get the corners of the right image in the left image
    corners2 = cv.perspectiveTransform(corners2_tmp, homography_matrix)

    # Get the bounding box
    bounding_box = np.concatenate((corners1, corners2), axis=0)

    # Get the minimum and maximum x and y coordinates
    [min_x, min_y] = np.int32(bounding_box.min(axis=0).ravel() - 0.5)
    [max_x, max_y] = np.int32(bounding_box.max(axis=0).ravel() + 0.5)

    # Get the translation matrix
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Warp the images
    # result = cv.warpPerspective(image2, translation_matrix.dot(homography_matrix), (image1.shape[1] + image2.shape[2], image1.shape[0]))
    result = cv.warpPerspective(image2, translation_matrix.dot(homography_matrix), (max_x - min_x, max_y - min_y))
    result[-min_y:rows1 - min_y, -min_x:cols1 - min_x] = image1
    # result[0:rows1, 0:cols1] = image1

    return result

# def __warp_images(
#         image1: np.ndarray,
#         image2: np.ndarray,
#         homography_matrix: np.ndarray,
#         ) -> np.ndarray:
#     """
#     This function is used to warp the images.
    
#     Parameters:
#     -----------
#     :type image1: numpy.ndarray
#     :type image2: numpy.ndarray
#     :type homography_matrix: numpy.ndarray
    
#     Returns:
#     --------
#     :rtype: numpy.ndarray
#     """

#     result = cv.warpPerspective(image1, homography_matrix, (image1.shape[1] + image2.shape[1], image1.shape[0]))
#     result[0:image2.shape[0], 0:image2.shape[1]] = image2

#     return result