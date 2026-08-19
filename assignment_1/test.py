import cv2
import numpy as np

print(cv2.__version__)
print(np.__version__)

def print_image_information(image):
    print("Image height:", image.shape[0])
    print("Image width:", image.shape[1])
    print("Image channels:", image.shape[2] if len(image.shape) > 2 else 1)
    print("Image size:", image.size)
    print("Image data type:", image.dtype)

image = cv2.imread("iris-1.jpg")
print_image_information(image)

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = cam.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))



with open('./assignments/camera_info.txt', 'w') as f:
    f.write(f"Frame width: {frame_width}\n")
    f.write(f"Frame height: {frame_height}\n")
    f.write(f"Frame FPS: {frame_fps}\n")

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()