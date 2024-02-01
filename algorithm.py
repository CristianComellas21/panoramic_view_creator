import cv2 as cv
import numpy as np
from tqdm import tqdm
from geometry import line, intersection

MIN_GOOD_MATCHES = 20
MIN_RATIO = 0.4
MAX_DESCRIPTORS = 15000
BASE_DESCRIPTION = "Panoramic view creation"

DETECTOR_OPTIONS = {
    "SIFT": cv.SIFT_create,
    "ORB": cv.ORB_create
}


def create_panoramic_view(
        images: [np.ndarray],
        match_ratio: float = 0.6,
        crop: bool = False,
        detector: str = "SIFT",
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

    # Assertions
    assert len(images) == 3, "The number of images must be 3."
    assert detector in DETECTOR_OPTIONS.keys(), f"Detector must be one of {DETECTOR_OPTIONS.keys()}."
    assert match_ratio > MIN_RATIO and match_ratio <= 1, "Match threshold must be between 0 and 1."
    


    # Create progress indicator
    progress_indicator = tqdm(desc=BASE_DESCRIPTION, bar_format="{desc} | Elapsed: {elapsed}") if verbose else None
        
    __update_progress(progress_indicator, "Preparing images...")

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
        
    __update_progress(progress_indicator, "Getting descriptors and key points...")

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
     ) =  __get_descriptors_and_key_points([new_left_image_gray, new_middle_image_gray, middle_image_gray, right_image_gray], detector=detector)
    
        
    __update_progress(progress_indicator, "Getting matches...")
    

    # Get the matches between consecutive images
    matches_left_middle = __get_matches(descriptors_new_middle, descriptors_new_left)
    matches_middle_right = __get_matches(descriptors_right, descriptors_middle)

        
    __update_progress(progress_indicator, "Getting good matches...")

    # Get the good matches
    good_matches_left_middle = __filter_matches(matches_left_middle, match_ratio)
    good_matches_middle_right = __filter_matches(matches_middle_right, match_ratio)

    
    __update_progress(progress_indicator, "Getting homography matrices...")

    # Get the homography matrix for each pair of consecutive images
    homography_matrix_left_middle = __get_homography_matrix(good_matches_left_middle, key_points_new_middle, key_points_new_left)
    homography_matrix_middle_right = __get_homography_matrix(good_matches_middle_right, key_points_right, key_points_middle)


    # Warp the images    
    height = middle_image.shape[0]

        
    __update_progress(progress_indicator, "Warping images left middle...")

    # Warp left and middle flipped
    result_left_middle = cv.warpPerspective(new_middle_image, homography_matrix_left_middle, (left_image.shape[1] + middle_image_gray.shape[1], height))
    result_left_middle[0:height, 0:new_left_image.shape[1]] = new_left_image
        
    
    __update_progress(progress_indicator, "Warping images right middle...")

    # Warp right and middle
    result_right_middle = cv.warpPerspective(right_image, homography_matrix_middle_right, (middle_image_gray.shape[1] + right_image.shape[1], height))
    result_right_middle[0:height, 0:middle_image.shape[1]] = middle_image
    

    ### === CROP THE IMAGES === ###


    if crop:
        
        __update_progress(progress_indicator, "Cropping images...")
        
        
        result_left_middle, result_right_middle = __crop_images(
            result_left_middle=result_left_middle,
            result_right_middle=result_right_middle,
            right_image=right_image,
            new_middle_image=new_middle_image,
            new_left_image=new_left_image,
            homography_matrix_left_middle=homography_matrix_left_middle,
            homography_matrix_middle_right=homography_matrix_middle_right
        )


    ### === JOIN THE IMAGES === ###
        
        
    __update_progress(progress_indicator, "Joining images...")
    
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

        
    __update_progress(progress_indicator, "Done!")


    return result_left_middle, result_right_middle, final_image
    
    
    

### ============== PRIVATE FUNCTIONS ============== ###

### --- IMAGE PROCESSING --- ###

def __get_descriptors_and_key_points(
    images: [np.ndarray],
    detector: str 
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
    detector = DETECTOR_OPTIONS.get(detector)(MAX_DESCRIPTORS)

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


def __crop_images(*, result_left_middle, result_right_middle, right_image, new_middle_image, new_left_image, homography_matrix_left_middle, homography_matrix_middle_right):
    """
    Crop the images to remove the black parts.
    
    Parameters:
    -----------
    :type result_left_middle: numpy.ndarray
    :type result_right_middle: numpy.ndarray
    :type right_image: numpy.ndarray
    :type new_middle_image: numpy.ndarray
    :type new_left_image: numpy.ndarray
    :type homography_matrix_left_middle: numpy.ndarray
    :type homography_matrix_middle_right: numpy.ndarray
    
    Returns:
    --------
    :rtype: (numpy.ndarray, numpy.ndarray)
    
    
    """
    
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
    right_limit_line_equation = line(right_limit_points[0], right_limit_points[1])

    # - BOTTOM CROP

    # Select the interesting points
    bottom_left_new_middle = corners_new_middle[1][0]
    bottom_right_new_middle = corners_new_middle[2][0]
    bottom_left_right = corners_right[1][0]
    bottom_right_right = corners_right[2][0]


    # Create the line equations
    line_equation_new_middle = line(bottom_left_new_middle, bottom_right_new_middle)
    line_equation_right = line(bottom_left_right, bottom_right_right)

    # Create the list of candidates, which are the most right ones and 
    # the intersection points of the bottom line on the right image
    # and the vertical right line of the left image
    bottom_candidates = [
        bottom_right_right,
        bottom_right_new_middle,
        intersection(line_equation_new_middle, right_limit_line_equation),
        intersection(line_equation_right, right_limit_line_equation)
    ]

    # Now we want to obtain the point with the lowest value of y
    bottom_candidates = np.array(bottom_candidates)
    y_values = bottom_candidates[:, 1]
    y_values = np.clip(y_values, 0, result_left_middle.shape[0])
    bottom_limit = np.min(y_values)

    # - TOP CROP

    # Select the interesting points
    top_left_new_middle = corners_new_middle[0][0]
    top_right_new_middle = corners_new_middle[3][0]
    top_left_right = corners_right[0][0]
    top_right_right = corners_right[3][0]


    # Create the line equations
    line_equation_new_middle = line(top_left_new_middle, top_right_new_middle)
    line_equation_right = line(top_left_right, top_right_right)

    # Create the list of candidates, which are the most right ones and
    # the intersection points of the top line on the right image
    # and the vertical right line of the left image

    top_candidates = [
        top_right_right,
        top_right_new_middle,
        intersection(line_equation_new_middle, right_limit_line_equation),
        intersection(line_equation_right, right_limit_line_equation)
    ]


    # Now we want to obtain the point with the highest value of y
    top_candidates = np.array(top_candidates)
    y_values = top_candidates[:, 1]
    y_values = np.clip(y_values, 0, np.inf)
    top_limit = np.max(y_values)

    # - CROP

    # Crop the images
    result_left_middle = result_left_middle[int(top_limit):int(bottom_limit), 0:right_limit_new_middle]
    result_right_middle = result_right_middle[int(top_limit):int(bottom_limit), 0:right_limit_right]

    return result_left_middle, result_right_middle


### --- PROGRESS INDICATOR --- ###

def __update_progress(progress_indicator: tqdm, mssg: str):
    """
    This function is used to update the progress indicator.

    Parameters:
    -----------
    :type progress_indicator: tqdm
    :type mssg: str

    Returns:
    --------
    :rtype: None
    """
    if progress_indicator is not None:
        progress_indicator.set_description_str(f"{BASE_DESCRIPTION} | {mssg}")