import cv2
import sys
import numpy as np
import os
import glob



def nothing(x):
    pass


for i in range(748):
    # Load in image
    image = cv2.imread('input/'+str(i)+'.jpg')

    # new for video
    # cap = cv2.VideoCapture("./inputVideo.mp4")
    # fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = frame_count/fps

    # print('fps = ' + str(fps))
    # print('number of frames = ' + str(frame_count))
    # print('duration (S) = ' + str(duration))
    # minutes = int(duration/60)
    # seconds = duration % 60
    # print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    # cap.release()


    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    cv2.setTrackbarPos('VMin', 'image', 216)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    # print("the code is", os.path.abspath(os.getcwd()))

    print("converting Image"+ str(i))

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

# cv2.imwrite(os.path.abspath(os.getcwd())+'result', output)

    cv2.imwrite("result/result"+str(i)+".jpg", output)

# Print if there is a change in HSV value
# if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
#         print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
#             hMin, sMin, vMin, hMax, sMax, vMax))
#         phMin = hMin
#         psMin = sMin
#         pvMin = vMin
#         phMax = hMax
#         psMax = sMax
#         pvMax = vMax

# Display output image
    cv2.imshow('image', output)

# Wait longer to prevent freeze for videos.
# if cv2.waitKey(wait_time) & 0xFF == ord('q'):
#     break

    cv2.destroyAllWindows()


# import cv2
# import numpy as np

 
img_array = []
for i in range(748):
    img = cv2.imread('result/result'+str(i)+'.jpg')
    # img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'),15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
