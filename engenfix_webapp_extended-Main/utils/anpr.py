# automated number plate recognition is done using image processing techniques
# in the future this will be built as a computer vision model

# importing libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


# reading the image and preprocessing
def image_preprocess(file):

    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

    # apply edge filter and find edges for localization
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    # find contours and apply the mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # use easyocr to read the number plate cropped_image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    # if number plate cannot be read
    if len(result):
        vehicle_number = result[0][-2]
    else:
        vehicle_number = 'Not Readable/Missing'

    
    return vehicle_number