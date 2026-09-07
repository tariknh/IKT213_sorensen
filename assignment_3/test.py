import os

import cv2
import numpy as np

# Anchors every path to this file's folder, so the script works no matter
# which directory you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))


def sobel_edge_detection(image):
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=1, ksize=1)

    # Sobel output is signed and can exceed 255, so scale it back to 8-bit.
    return cv2.convertScaleAbs(sobel)


def canny_edge_detection(image, threshold_1, threshold_2):
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(gray, threshold_1, threshold_2)


def template_match(image, template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = gray_template.shape

    # TM_CCOEFF_NORMED scores 1.0 for a perfect match, so a high threshold
    # keeps only the good hits.
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    matches = image.copy()
    for pt in zip(*np.where(result >= 0.9)[::-1]):
        cv2.rectangle(matches, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    return matches


def resize(image, scale_factor: int, up_or_down: str):
    # pyrUp/pyrDown only ever step by a factor of 2, so scale_factor is the
    # number of pyramid levels to climb, applied one step at a time.
    if up_or_down == "up":
        for _ in range(scale_factor):
            image = cv2.pyrUp(image)
    elif up_or_down == "down":
        for _ in range(scale_factor):
            image = cv2.pyrDown(image)
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    return image


lambo = cv2.imread(os.path.join(HERE, "lambo.png"))
shapes = cv2.imread(os.path.join(HERE, "shapes-1.png"))
shapes_template = cv2.imread(os.path.join(HERE, "shapes_template.jpg"))

cv2.imwrite(os.path.join(HERE, "image_sobel.png"), sobel_edge_detection(lambo))
cv2.imwrite(os.path.join(HERE, "image_canny.png"), canny_edge_detection(lambo, 50, 50))
cv2.imwrite(os.path.join(HERE, "image_template.png"), template_match(shapes, shapes_template))
cv2.imwrite(os.path.join(HERE, "image_resized_up.png"), resize(lambo, 2, "up"))
cv2.imwrite(os.path.join(HERE, "image_resized_down.png"), resize(lambo, 2, "down"))
