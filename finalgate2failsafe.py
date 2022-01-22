#from nms import non_max_suppression_fast
import numpy as np
import cv2
import imutils
import time

cap = cv2.VideoCapture("gate12:5-2.mp4")

while(cap.isOpened()):
        
	ret, frame = cap.read()
	if ret == True:

		frame = frame[0:1100, 300:2704]

		height = np.size(frame)
		width = np.size(frame)


		cap_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# define the list of boundaries
		# lower mask (0-10)
		lower_red = np.array([0,20,40])
		upper_red = np.array([55,255,255])
		mask0 = cv2.inRange(cap_hsv, lower_red, upper_red)
		                      									#for red color
		# upper mask (170-180)
		lower_red = np.array([140,0,0])
		upper_red = np.array([185,255,255])
		mask1 = cv2.inRange(cap_hsv, lower_red, upper_red)

		# join my masks
		mask = mask0 + mask1

		output = cv2.bitwise_and(frame, frame, mask = mask)
		 
			# show the images
			#find contours 
			#instead of showing image on screen print x coordinages, convert into percentage, find width of gate 
		#cv2.imshow("frames", np.hstack([frame, output]))

		
		#convert images to grayscale
		gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5,5), 0)
		gray_blur = cv2.Canny(gray, 2,15)
		cv2.imshow("blur", gray_blur)
		cv2.waitKey(1)

		mask = gray.copy()
		kernel = np.ones((5,5),np.uint8)
		eroded = cv2.erode(mask, kernel, iterations=4)
		#cv2.imshow("eroded", eroded)
		#cv2.waitKey(0)

		#Erosions and dilations
		#erosions are apploed to reduce the size of foreground objects
		mask = gray.copy()
		kernel = np.ones((5,5),np.uint8)
		dilated = cv2.dilate(eroded, kernel, iterations=10)
		cv2.imshow("dilated", dilated)
		cv2.waitKey(1)

		#cv.Mat vesselImage = cv.imread(mask)


		#edge detection
		#applying edge detection 
		edged = cv2.Canny(dilated, 30,150)
		#cv2.imshow("Edged", edged)
		#cv2.waitKey(0)

		#detecting and drawing countours
		#find contours(outlines) of the foreground objects in the thresholded image
		cnts, heirarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		#cv2.drawContours(immat,contours,-1,CV_RGB(255,0,0),2);
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:2]

		

		boundingBoxes = np.empty((0, 4), float)

		for c in cnts:
			#approcimate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			for x in range(len(approx)):
				cv2.circle(frame, (approx[x][0][0], approx[x][0][1],), 7, (0, 0, 255), -1)
			#cv2.Circle(image, (approx[1][0][0], approx[1][0][1],), 7, (0, 0, 255), -1)

			#print(approx)


			x,y,w,h = cv2.boundingRect(c)

			#boundingBoxes.append(np.array[x,y,w,h])
			#boundingBoxes = np.append(boundingBoxes, np.array([[x,y,x+w,y+h]]), axis = 0)
			#boundingBoxes = np.append(boundingBoxes, np.array([[h,w,y,x]]), axis = 0)
			box = cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)
			##cv2.imshow("bounding rectangle",frame)
			##cv2.waitKey(1)
			#time.sleep(0.1)

			#print(str(x/width) + " " + str(y/height) + " " + str((x+w)/width) + " " +  str((y+h)/height))
			#print (x , " " , y , " " , x+w , " " ,  y+h)
			area = w*h
			ratio = h/w
			#print("w: " + str(w) + " h: " + str(h) + " a:" + str(area))
			#print(h/w)
		

			if(ratio>2):
				cv2.imshow("bounding rectangle",box)
				cv2.waitKey(1)
				print(x, x+w, y, y+w, area)
				#print(h)
			else:
				print("false")

			


cap.release()
cv2.destroyAllWindows()