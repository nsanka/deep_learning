# Import Modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    # Convert Color image to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce Noise by applying Blur using 5x5 kernel on the image pixels
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Identify the lines using Canny method
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Returns the start and end coordinates of the line using slope and intercept
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # Assume line starts at the image bottom
    y1 = image.shape[0]
    # Assume line ends at 3/5 of image height
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# Returns two lines with average slope and intercept
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    # Separate two lines based on the slope of each line
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # Calculate average slope and intercept
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # Get the coordinates based on the slope and intercept
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def display_lines(image, lines):
    # Define an image array of same shape as image
    line_image = np.zeros_like(image)
    if lines is not None:
        # Append each line in lines to image array
        for line in lines:
            # Unpack two dimensional line
            x1, y1, x2, y2 = line.reshape(4)
            # Add line to image array
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    # Get height of the image
    height = image.shape[0]
    # Define area of interest as an array with 1 triangle coordinates
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    # Define an image with zerios
    mask = np.zeros_like(image)
    # Fill the area of triangle in the mask with white pixles
    cv2.fillPoly(mask, polygons, 255)
    # Apply bitwise & to keep only region of interest in image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def show_lines_in_image(image, ms=1):
    # Copy Image
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    # Get region of interest
    cropped_image = region_of_interest(canny_image)
    # Grab the lines using Hough Lines method 2 rows, 1 radian, 100 threshold
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)
    # Calculate average slope and intercept for the two lines
    averaged_lines = average_slope_intercept(lane_image, lines)
    # Get all lines as image
    line_image = display_lines(lane_image, averaged_lines)
    # Add lines image to the original image
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # Open Image in a window
    cv2.imshow('result', combo_image)
    # Wait until we press 'q' or milliseconds (bitwise & 0xFF masks the integer to 8bit)
    return (cv2.waitKey(ms) & 0xFF == ord('q'))

# Read image as numpy array
#image = cv2.imread('road_image.jpg')
#show_lines_in_image(image, 0)
#exit(0)

cap = cv2.VideoCapture('road_video.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    if show_lines_in_image(frame):
        break

# Release the Video
cap.release()
# Destroy all cv2 windows
cv2.destroyAllWindows()