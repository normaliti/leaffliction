import cv2

def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (15,15), 0)
    return blurred_image

def create_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]

