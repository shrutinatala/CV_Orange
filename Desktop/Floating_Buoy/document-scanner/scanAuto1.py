import cv2
import numpy as np
import math
                                                                                                                                                                                                                                                                   
video = cv2.VideoCapture("buoyOfficial.mp4")
#video = cv2.VideoCapture("gateB.mp4")
#video = cv2.VideoCapture(0)
i = 0
# Downscale the image to a reasonable size to reduce compute
scale = 1

# Minimize false detects by eliminating contours less than a percentage of the image
area_threshold = 0.1

ret, orig_frame = video.read()
width = orig_frame.shape[0]
height = orig_frame.shape[1]
dim = (int(scale*height), int(scale*width))

while (True):
  ret, orig_frame = video.read()
  if not ret:
    break
  print(i)
  i = i+1

  orig_frame = cv2.resize(orig_frame, dim, interpolation = cv2.INTER_AREA)
  frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
  #frame = cv2.convertScaleAbs(frame, frame, 2.0, 60)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
  mask = cv2.inRange(hsv,(0, 0, 176.5), (180, 255, 225))
  #mask1 = cv2.inRange(hsv, (), (90, ))
  '''mask0 = cv2.inRange(hsv, (0, 0, 0), (0, 0, 79))
  mask1 = cv2.inRange(hsv, (180, 255, 150), (255, 255, 255))
  mask = mask0 + mask1'''

  cv2.imshow('Mask', mask)
  #cv2.waitKey(0)

  ret, thresh = cv2.threshold(mask, 127, 255,0)
    #Erosions and dilations
    #erosions are apploed to reduce the size of foreground objects
  kernel = np.ones((3,3),np.uint8)
  eroded = cv2.erode(thresh, kernel, iterations=1)  
  dilated = cv2.dilate(eroded, kernel, iterations=1)
  #cv2.imshow("dilated", dilated)
  #cv2.imshow("Edged", edged)

  cnts,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #cv2.drawContours(orig_frame, cnts, -1, (0, 255, 0), 3)
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]

  boundingBoxes = np.empty((0, 4), float)
  if len(cnts) > 0: 
    cnt = cnts[0]
    area = cv2.contourArea(cnt)
    if area/(dim[0]*dim[1]) > area_threshold:

      M = cv2.moments(cnts[0])
      for c in cnts:
        rect = cv2.minAreaRect(c)
        #print("rect: {}".format(rect))

        # the order of the box points: bottom left, top left, top right,
        # bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #print("bounding box: {}".format(box))
        cv2.drawContours(orig_frame, [box], 0, (0, 0, 255), 2)
        #x,y,w,h = cv2.boundingRect(c)

        #boundingBoxes = np.append(boundingBoxes, np.array([[x,y,x+w,y+h]]), axis = 0)
        #cv2.rectangle(orig_frame,(x,y), (x+w, y+h), (255,0,0), 2)

        #print(str(x/width) + " " + str(y/height) + " " + str((x+w)/width) + " " +  str((y+h)/height))

        print(box)
        print("hi")
        '''if M["m00"] != 0:
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
        else:
          cX, cY = 0,0

      print(cX/width, cY/height)'''
  print("=========================================")

  
  cv2.imshow("bounding rectangle",orig_frame)

  key = cv2.waitKey(1)
  if key == 27:
    break

video.release()
cv2.destroyAllWindows()


