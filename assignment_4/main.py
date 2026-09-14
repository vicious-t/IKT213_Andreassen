import cv2
import numpy as np


def save_image(filename, image):
    success = cv2.imwrite(filename, image)
    if not success:
        print(f"Uh oh, could not save {filename}")


def grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def harris_corner_detection(reference_image):
    grayscale_image = np.float32(grayscale(reference_image))

    harris_result = cv2.cornerHarris(
        grayscale_image,
        2,
        3,
        0.04
    )

    harris_result = cv2.dilate(
        harris_result,
        None
    )

    harris_image = reference_image.copy()

    harris_image[
        harris_result > 0.01 * harris_result.max()
        ] = [0, 0, 255]

    return harris_image


def find_sift_features(image):
    grayscale_image = grayscale(image)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        grayscale_image,
        None
    )

    return keypoints, descriptors


def match_sift_features(
        descriptors_1,
        descriptors_2,
        good_match_percent
):
    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        descriptors_1,
        descriptors_2,
        k=2
    )

    good_matches = []

    for match_1, match_2 in matches:
        if match_1.distance < good_match_percent * match_2.distance:
            good_matches.append(match_1)

    return good_matches


def get_matched_points(
        keypoints_1,
        keypoints_2,
        matches
):
    points_1 = np.float32(
        [keypoints_1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    points_2 = np.float32(
        [keypoints_2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    return points_1, points_2


def align_images(
        image_to_align,
        reference_image,
        min_match_count,
        good_match_percent
):
    keypoints_1, descriptors_1 = find_sift_features(
        image_to_align
    )

    keypoints_2, descriptors_2 = find_sift_features(
        reference_image
    )

    good_matches = match_sift_features(
        descriptors_1,
        descriptors_2,
        good_match_percent
    )

    if len(good_matches) < min_match_count:
        raise ValueError(
            f"Uh oh, not enough matches found: "
            f"{len(good_matches)}/{min_match_count}"
        )

    points_1, points_2 = get_matched_points(
        keypoints_1,
        keypoints_2,
        good_matches
    )

    homography, mask = cv2.findHomography(
        points_1,
        points_2,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        raise ValueError("Uh oh, could not calculate homography")

    matches_image = cv2.drawMatches(
        image_to_align,
        keypoints_1,
        reference_image,
        keypoints_2,
        good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask.ravel().tolist(),
        flags=2
    )

    height, width = reference_image.shape[:2]

    aligned_image = cv2.warpPerspective(
        image_to_align,
        homography,
        (width, height)
    )

    return aligned_image, matches_image


def main():
    reference_image = cv2.imread("images/reference_img.png")
    if reference_image is None:
        print("Uh oh, could not open reference_img.png")
        return

    image_to_align = cv2.imread("images/align_this.jpg")
    if image_to_align is None:
        print("Uh oh, could not open align_this.jpg")
        return

    # Harris corner detection
    harris_image = harris_corner_detection(reference_image)
    save_image("solutions/harris.png", harris_image)

    # SIFT feature matching
    aligned_image, matches_image = align_images(
        image_to_align,
        reference_image,
        10,
        0.7
    )

    save_image("solutions/aligned.png", aligned_image)
    save_image("solutions/matches.png", matches_image)


if __name__ == "__main__":
    main()
