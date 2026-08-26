import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

print(cv2.__version__)
print(np.__version__)

def padding(image,border_width):
    return cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)

def crop(image, x_0, x_1,  y_0, y_1):
    return image[y_0:y_1, x_0:x_1]


def resize(image, width, height):
    return cv2.resize(image, (width, height))

def copy(image,emptyPictureArray):
    np.copyto(emptyPictureArray, image)
    return emptyPictureArray

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image,emptyPictureArray,hue):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                emptyPictureArray[i][j][k] = (int(image[i][j][k]) + hue) % 256
    return emptyPictureArray

def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

image = cv2.imread(os.path.join(HERE, "iris-1.jpg"))
imageWidth = image.shape[1]
imageHeight = image.shape[0]
imageChannels = image.shape[2] if len(image.shape) > 2 else 1
emptyPictureArrayCopied = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
emptyPictureArrayHue = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)

if image is None:
    raise FileNotFoundError("Could not read iris-1.jpg")

borderImage = padding(image, 100)
croppedImage = crop(image, 200, 670, 200, 470)
resizedImage = resize(image, 200, 200)
copiedImage = copy(image, emptyPictureArrayCopied)
grayscaleImage = grayscale(image)
hsvImage = hsv(image)
hueShiftedImage = hue_shifted(image, emptyPictureArrayHue, 50)
smoothingImage = smoothing(image)
rotatedImage = rotation(image, 180)
cv2.imwrite(os.path.join(HERE, "iris-1-border.jpg"), borderImage)
cv2.imwrite(os.path.join(HERE, "iris-1-cropped.jpg"), croppedImage)
cv2.imwrite(os.path.join(HERE, "iris-1-resized.jpg"), resizedImage)
cv2.imwrite(os.path.join(HERE, "iris-1-copied.jpg"), copiedImage)
cv2.imwrite(os.path.join(HERE, "iris-1-grayscale.jpg"), grayscaleImage)
cv2.imwrite(os.path.join(HERE, "iris-1-hsv.jpg"), hsvImage)
cv2.imwrite(os.path.join(HERE, "iris-1-hue-shifted.jpg"), hueShiftedImage)
cv2.imwrite(os.path.join(HERE, "iris-1-smoothing.jpg"), smoothingImage)
cv2.imwrite(os.path.join(HERE, "iris-1-rotated.jpg"), rotatedImage)


cv2.destroyAllWindows()