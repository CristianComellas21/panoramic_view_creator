import cv2 as cv
import numpy as np
from tqdm import tqdm

MIN_GOOD_MATCHES = 10
MIN_RATIO = 0.4
MAX_DESCRIPTORS = 15000
BASE_DESCRIPTION = "Panoramic view creation"

def create_panoramic_view(
        images: [np.ndarray],
        match_threshold: float = 0.6,
        verbose: bool = True
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

    assert len(images) == 3, "The number of images must be 3."


    # Create progress indicator
    progress_indicator = tqdm(desc=BASE_DESCRIPTION, bar_format="{desc} | Elapsed: {elapsed}") if verbose else None

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Preparing images...")

    # Take the images
    left_image = images[0]
    middle_image = images[1]
    right_image = images[2]
    new_middle_image = cv.flip(left_image, 1)
    new_left_image = cv.flip(middle_image, 1)
    
    # Convert the images to grayscale
    middle_image_gray = cv.cvtColor(middle_image, cv.COLOR_BGR2GRAY)
    right_image_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    new_middle_image_gray = cv.cvtColor(new_middle_image, cv.COLOR_BGR2GRAY)
    new_left_image_gray = cv.cvtColor(new_left_image, cv.COLOR_BGR2GRAY)

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Getting descriptors and key points...")

    # Get the descriptors and key points of the images
    (
        descriptors_new_left, 
        descriptors_new_middle,
        descriptors_middle,
        descriptors_right
    ), \
    (
        key_points_new_left,
        key_points_new_middle,
        key_points_middle,
        key_points_right
     ) =  __get_descriptors_and_key_points([new_left_image_gray, new_middle_image_gray, middle_image_gray, right_image_gray])
    
    
    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Getting matches...")

    # Get the matches between consecutive images
    matches_left_middle = __get_matches(descriptors_new_middle, descriptors_new_left)
    matches_middle_right = __get_matches(descriptors_right, descriptors_middle)

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Getting good matches...")

    # Get the good matches
    good_matches_left_middle = __filter_matches(matches_left_middle, match_threshold)
    good_matches_middle_right = __filter_matches(matches_middle_right, match_threshold)

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Getting homography matrices...")

    # Get the homography matrix for each pair of consecutive images
    homography_matrix_left_middle = __get_homography_matrix(good_matches_left_middle, key_points_new_middle, key_points_new_left)
    homography_matrix_middle_right = __get_homography_matrix(good_matches_middle_right, key_points_right, key_points_middle)


    # Warp the images    
    height = middle_image.shape[0]

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Warping images left middle...")

    # Warp left and middle flipped
    result_left_middle = cv.warpPerspective(new_middle_image, homography_matrix_left_middle, (left_image.shape[1] + middle_image_gray.shape[1], height))
    result_left_middle[0:height, 0:new_left_image.shape[1]] = new_left_image


    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Warping images right middle...")

    # Warp right and middle
    result_right_middle = cv.warpPerspective(right_image, homography_matrix_middle_right, (middle_image_gray.shape[1] + right_image.shape[1], height))
    result_right_middle[0:height, 0:middle_image.shape[1]] = middle_image
    





    ### === CROP THE IMAGES === ###


    # Get corners of the new_middle image into the new image
    corners_new_middle_tmp = np.float32([[0, 0], [0, new_middle_image.shape[0]], [new_middle_image.shape[1], new_middle_image.shape[0]], [new_middle_image.shape[1], 0]]).reshape(-1, 1, 2)
    corners_new_middle = cv.perspectiveTransform(corners_new_middle_tmp, homography_matrix_left_middle)


    # --- CROP HORITZONTALLY LEFT MIDDLE

    # Get x corners and take the second rightmost one
    x_values = corners_new_middle[:, :, 0].squeeze()
    x_values = np.sort(x_values)
    right_limit_new_middle = int(x_values[2])



    # Get corners of the right image into the new image
    corners_right_tmp = np.float32([[0, 0], [0, right_image.shape[0]], [right_image.shape[1], right_image.shape[0]], [right_image.shape[1], 0]]).reshape(-1, 1, 2)
    corners_right = cv.perspectiveTransform(corners_right_tmp, homography_matrix_middle_right)

    # --- CROP HORIZONTALLY RIGHT MIDDLE

    # Get x corners and take the second rightmost one
    x_values = corners_right[:, :, 0].squeeze()
    x_values = np.sort(x_values)
    right_limit_right = int(x_values[2])


    # --- CROP VERTICALLY BOTH IMAGES

    # Get the points of the right limit of the most left image
    right_limit_points = np.array([
        [new_left_image.shape[1], 0],
        [new_left_image.shape[1], new_left_image.shape[0]]
    ])

    # Get the equation of the line that goes through the right limit points
    right_limit_line_equation = __line(right_limit_points[0], right_limit_points[1])

    # - BOTTOM CROP

    # Select the interesting points
    bottom_left_new_middle = corners_new_middle[1][0]
    bottom_right_new_middle = corners_new_middle[2][0]
    bottom_left_right = corners_right[1][0]
    bottom_right_right = corners_right[2][0]


    # Create the line equations
    line_equation_new_middle = __line(bottom_left_new_middle, bottom_right_new_middle)
    line_equation_right = __line(bottom_left_right, bottom_right_right)

    # Create the list of candidates, which are the most right ones and 
    # the intersection points of the bottom line on the right image
    # and the vertical right line of the left image
    bottom_candidates = [
        bottom_right_right,
        bottom_right_new_middle,
        __intersection(line_equation_new_middle, right_limit_line_equation),
        __intersection(line_equation_right, right_limit_line_equation)
    ]

    print(f"bottom candidates: {bottom_candidates}")

    # Now we want to obtain the point with the lowest value of y
    bottom_candidates = np.array(bottom_candidates)
    y_values = bottom_candidates[:, 1]
    bottom_limit = np.min(y_values)

    print(f"bottom limit: {bottom_limit}")

    # - TOP CROP

    # Select the interesting points
    top_left_new_middle = corners_new_middle[0][0]
    top_right_new_middle = corners_new_middle[3][0]
    top_left_right = corners_right[0][0]
    top_right_right = corners_right[3][0]


    # Create the line equations
    line_equation_new_middle = __line(top_left_new_middle, top_right_new_middle)
    line_equation_right = __line(top_left_right, top_right_right)

    # Create the list of candidates, which are the most right ones and
    # the intersection points of the top line on the right image
    # and the vertical right line of the left image

    top_candidates = [
        top_right_right,
        top_right_new_middle,
        __intersection(line_equation_new_middle, right_limit_line_equation),
        __intersection(line_equation_right, right_limit_line_equation)
    ]

    print(f"top candidates: {top_candidates}")


    # Now we want to obtain the point with the highest value of y
    top_candidates = np.array(top_candidates)
    y_values = top_candidates[:, 1]
    top_limit = np.max(y_values)

    print(f"top limit: {top_limit}")

    # - CROP

    # Crop the images
    result_left_middle = result_left_middle[int(top_limit):int(bottom_limit), 0:right_limit_new_middle]
    result_right_middle = result_right_middle[int(top_limit):int(bottom_limit), 0:right_limit_right]


    # # Bottom points of the new middle image
    # bottom_left = corners_new_middle[2][0]
    # bottom_right = corners_new_middle[1][0]

    # # cv.circle(result_left_middle, (int(bottom_left[0]), int(bottom_left[1])), 50, (0, 0, 255), -1)
    # # cv.circle(result_left_middle, (int(bottom_right[0]), int(bottom_right[1])), 50, (0, 0, 255), -1)

    # # Get the line equation
    # line_equation_top = __line(bottom_right, bottom_left)

    

    # # cv.line(result_left_middle, (int(right_limit_points[0][0]), int(right_limit_points[0][1])), (int(right_limit_points[1][0]), int(right_limit_points[1][1])), (0, 255, 0), 30)

    # # Get the line equation
    # right_limit_line_equation = __line(right_limit_points[0], right_limit_points[1])

    # # Get the intersection point
    # intersection_point = __intersection(line_equation_top, right_limit_line_equation)
    # print(f"intersection: {intersection_point}")

    

    # cv.circle(result_left_middle, (int(intersection_point[0]), int(intersection_point[1])), 50, (255, 0, 0), -1)









    


    ### === JOIN THE IMAGES === ###

    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Joining images...")
    
    # Flip the result before joining
    result_left_middle = cv.flip(result_left_middle, 1)

    # Calculate the width of the final image. The results have two times the width of the middle image, 
    # so we need to subtract the width of the middle image once
    width = result_left_middle.shape[1] + result_right_middle.shape[1] - middle_image.shape[1]
    height = result_left_middle.shape[0]

    # Create the final image
    final_image = np.zeros((height, width, 3), np.uint8)

    # Add the left result to the final image (Includes left and middle images)
    final_image[:, 0:result_left_middle.shape[1]] = result_left_middle

    # Add the right result to the final image (Includes middle and right images)
    final_image[:, result_left_middle.shape[1]:result_left_middle.shape[1]+right_image.shape[1]] = result_right_middle[:, middle_image.shape[1]:middle_image.shape[1]+right_image.shape[1]]



    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | Done!")
        progress_indicator.close()


    # # Get corners of the image1 into the new image 
    # corners1_tmp = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    # corners1 = cv.perspectiveTransform(corners1_tmp, homography_matrix)


    # # Get corners of the image2 into the new image (this doesn't change)
    # corners2 = np.float32([[0, 0], [0, image2.shape[0]], [image2.shape[1], image2.shape[0]], [image2.shape[1], 0]]).reshape(-1, 1, 2)

    # # Get y boundaries
    # y_values = np.concatenate((corners1, corners2), axis=0)[:, :, 1].squeeze()
    # y_values = np.sort(y_values)
    # upper_limit = int(y_values[3])
    # lower_limit = int(y_values[4])

    # print(upper_limit, lower_limit)

    # # Get x boundaries
    # x_values = corners1[:, :, 0].squeeze()
    # x_values = np.sort(x_values)
    # right_limit = int(x_values[2])



    return result_left_middle, result_right_middle, final_image
    
### Private Functions ###

def __get_descriptors_and_key_points(
    images: [np.ndarray],
) -> (np.ndarray, np.ndarray):
    """
    This function is used to get the descriptors and key points of the images.

    Parameters:
    ----------
    :type images: list

    Returns:
    --------
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    # Create the descritor detector object (SIFT in this case)
    detector = cv.SIFT_create(MAX_DESCRIPTORS)

    # Get the descriptors and key points of the images
    descriptors_list = []
    key_points_list = []
    for image in images:
        key_points, descriptors = detector.detectAndCompute(image, None)
        descriptors_list.append(descriptors)
        key_points_list.append(key_points)

    return descriptors_list, key_points_list


def __get_matches(
        descriptors_src: np.ndarray,
        descriptors_dst: np.ndarray,
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
    matcher = cv.BFMatcher_create()

    # Get the matches between consecutive images
    matches = matcher.knnMatch(descriptors_src, descriptors_dst, k=2)

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

    # Get only the matches that are close enough
    filtered_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * ratio:
            filtered_matches.append(m1)

    # Check if there are enough good matches
    n_good_matches = len(filtered_matches)
    if n_good_matches < MIN_GOOD_MATCHES:
        raise Exception(f"Not enough good matches. Only {n_good_matches} found and at least {MIN_GOOD_MATCHES} are needed. Please try with a higher match threshold.")

    return filtered_matches

def __get_homography_matrix(
        good_matches: [cv.DMatch],
        key_points_src: [cv.KeyPoint],
        key_points_dst: [cv.KeyPoint],
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
    points_src = np.float32([key_points_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_dst = np.float32([key_points_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Get the homography matrix
    homography_matrix, _ = cv.findHomography(points_src, points_dst, cv.RANSAC, 5.0)

    return homography_matrix



def __line(point1: np.array, point2: np.array):
    A = (point1[1] - point2[1])
    B = (point2[0] - point1[0])
    C = -(point1[0]*point2[1] - point2[0]*point1[1])
    return A, B, C

def __intersection(line1: tuple, line2: tuple):

    A1, B1, C1 = line1
    A2, B2, C2 = line2

    D  = A1 * B2 - B1 * A2
    Dx = C1 * B2 - B1 * C2
    Dy = A1 * C2 - C1 * A2

    if D == 0:
        return None

    x = Dx / D
    y = Dy / D
    return np.array([x, y])