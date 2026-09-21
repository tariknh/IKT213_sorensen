import os

import cv2
import numpy as np

# Anchors every path to this file's folder, so the script works no matter
# which directory you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))

def harris_corner_detection(image):
    marked = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    marked[dst > 0.01 * dst.max()] = [0, 0, 255]

    return marked

def feature_based_image_alignment(image_to_align,reference_image,max_features,good_match_precent):

    sift = cv2.SIFT_create(max_features)


    # Detect SIFT features and compute descriptors.
    keypoints1, descriptors1 = sift.detectAndCompute(image_to_align, None)
    keypoints2, descriptors2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test: keep a match only if it is clearly better than the
    # second-best candidate. good_match_precent is the ratio threshold.
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    # findHomography needs at least 4 point pairs to solve for the 8 unknowns.
    if len(good) < 4:
        print(f"Not enough good matches to align: {len(good)} (need 4).")
        return None, None

    # Extract location of good matches
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # RANSAC throws out the mismatches that survived the ratio test.
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    print(f"good matches: {len(good)}, RANSAC inliers: {int(mask.sum())}")

    # Warp image_to_align into the reference image's frame, so the result has
    # the reference's dimensions and its content lines up with it.
    height, width = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, h, (width, height))

    # Draw only the inlier matches, in green.
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                       matchesMask=matches_mask, flags=2)
    im_matches = cv2.drawMatches(image_to_align, keypoints1, reference_image,
                                 keypoints2, good, None, **draw_params)

    cv2.imwrite(os.path.join(HERE, "matches.png"), im_matches)
    cv2.imwrite(os.path.join(HERE, "aligned.png"), aligned)

    return aligned, im_matches


harris = harris_corner_detection(cv2.imread(os.path.join(HERE, "reference_image.png")))
cv2.imwrite(os.path.join(HERE, "harris.png"), harris)

# max_features=0 means "no cap" in SIFT_create. The assignment suggests 10, but
# that yields 0 matches on these images -- see NOTE.md for the measurements.
feature_based_image_alignment(
    cv2.imread(os.path.join(HERE, "align_this.jpg")),
    cv2.imread(os.path.join(HERE, "reference_image.png")),
    0,
    0.7,
)
